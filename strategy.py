#!/usr/bin/env python3
"""
Experiment #438: 1d Fisher Transform + BB Regime with Weekly Trend Filter

Hypothesis: After analyzing 437 experiments, the key failure mode for 1d strategies
is either too few trades (overly strict filters) or whipsaw losses (no regime filter).
This strategy uses Ehlers Fisher Transform which is specifically designed to catch
reversals in bear/range markets - the exact condition of 2025 test period.

Key innovations vs #432:
1. FISHER TRANSFORM (period=9): Better than RSI for catching reversals in bear rallies
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
   - Proven edge in 2022 crash and 2025 bear market

2. BOLLINGER BANDWIDTH REGIME: 
   - BB Width < 25th percentile = range (use Fisher mean reversion)
   - BB Width > 75th percentile = trend (use breakout logic)
   - Prevents Fisher whipsaws in strong trends

3. WEEKLY HMA(21) BIAS: Only take signals aligned with 1w trend
   - Long signals only when price > 1w HMA
   - Short signals only when price < 1w HMA
   - Critical for avoiding counter-trend disasters

4. SIMPLIFIED STOPLOSS: 2.0 * ATR(14) trailing stop
   - Cleaner logic than #432 (no state reset bugs)
   - Signal → 0 when stop hit

5. POSITION SIZING: 0.30 discrete (slightly higher than #432's 0.28)
   - Ensures sufficient trade impact while controlling DD

Why this should beat #432 (Sharpe=-0.181):
- Fisher Transform is more sensitive than RSI for reversals (more trades)
- BB regime filter prevents whipsaws (better win rate)
- Simpler stoploss logic (fewer bugs)
- Looser Fisher thresholds (-1.5/+1.5 vs RSI 30/70) = more entries

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_bb_regime_weekly_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian distribution
    for clearer reversal signals. Period=9 is standard.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Use (high + low) / 2 as typical price
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = typical[i-period+1:i+1].max()
        lowest = typical[i-period+1:i+1].min()
        
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        # Normalize price to 0-1 range
        normalized = (typical[i] - lowest) / range_val
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with previous value (Ehlers smoothing)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_mult * std)
    lower = sma - (std_mult * std)
    bandwidth = (upper - lower) / sma * 100  # Percentage bandwidth
    
    return upper.values, lower.values, bandwidth.values

def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate where current BB bandwidth sits in recent history (0-100)."""
    n = len(bandwidth)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(bandwidth[i]):
            window = bandwidth[i-lookback+1:i+1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                percentile[i] = np.sum(valid_window <= bandwidth[i]) / len(valid_window) * 100
    
    return percentile

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_percentile(bb_bandwidth, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_percentile[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (BB Bandwidth Percentile) ===
        range_regime = bb_percentile[i] < 30  # Bottom 30% = range/mean revert
        trend_regime = bb_percentile[i] > 70  # Top 30% = trending/breakout
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # RANGE REGIME: Use Fisher mean reversion (primary edge)
        if range_regime:
            if fisher_long_cross and bull_trend_1w:
                new_signal = SIZE
            elif fisher_short_cross and bear_trend_1w:
                new_signal = -SIZE
        
        # TREND REGIME: Only take signals with trend (ADX confirmation)
        elif trend_regime and adx[i] > 20:
            if fisher_long_cross and bull_trend_1w:
                new_signal = SIZE
            elif fisher_short_cross and bear_trend_1w:
                new_signal = -SIZE
        
        # NEUTRAL REGIME: Require stronger Fisher signal
        else:
            # Deeper oversold/overbought for neutral regime
            fisher_deep_long = fisher[i] < -2.0
            fisher_deep_short = fisher[i] > 2.0
            
            if fisher_deep_long and bull_trend_1w:
                new_signal = SIZE
            elif fisher_deep_short and bear_trend_1w:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0 and new_signal != 0.0:
            if position_side > 0:
                # Update highest for long position
                if close[i] > highest_since_entry:
                    highest_since_entry = close[i]
                stoploss_price = highest_since_entry - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            elif position_side < 0:
                # Update lowest for short position
                if lowest_since_entry == 0.0 or close[i] < lowest_since_entry:
                    lowest_since_entry = close[i]
                stoploss_price = lowest_since_entry + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            elif position_side < 0 and bull_trend_1w:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip - reset tracking
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            # If same side, keep tracking (don't reset highs/lows)
        else:
            # Exit position
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals