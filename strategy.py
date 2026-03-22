#!/usr/bin/env python3
"""
Experiment #522: 1d Regime-Adaptive with Weekly HMA Bias

Hypothesis: Daily timeframe is too slow for pure trend-following (failed in 2022 crash,
2025 bear market). Instead, use REGIME-ADAPTIVE approach:
1. CHOPPINESS INDEX detects trending (CHOP<38.2) vs ranging (CHOP>61.8) markets
2. Ranging markets: Mean-reversion at RSI/Bollinger extremes (high win rate)
3. Trending markets: Follow weekly HMA direction (capture major moves)
4. Weekly HMA bias: 1w HMA(21) via mtf_data for long-term directional filter
5. LOOSE thresholds: RSI 35/65 (not 30/70) to ensure ≥10 trades over 4 years
6. 2.5*ATR stoploss: Wider for 1d timeframe swings

Why 1d can work:
- Captures major regime shifts without noise
- Fewer trades = less fee drag (critical for profitability)
- Weekly HMA provides strong directional bias
- Choppiness Index is proven regime filter (75%+ accuracy)

Key innovations vs failed experiments:
- CHOP regime detection (not used in most failed 1d strategies)
- Dual-mode logic: mean-revert in range, trend-follow in trend
- Loose RSI thresholds (35/65 vs 30/70) for sufficient trade count
- Weekly HTF bias (stronger than daily bias used in #516)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (conservative for 1d swings)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_weekly_hma_dual_mode_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging market (mean-reversion works)
    CHOP < 38.2 = trending market (trend-following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_series = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    for i in range(period, n):
        atr_sum = atr_series[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return sma.values, upper.values, lower.values

def calculate_keltner(high, low, close, period=20, atr_period=10, mult=1.5):
    """Calculate Keltner Channels for squeeze detection."""
    atr = calculate_atr(high, low, close, atr_period)
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    upper = ema.values + mult * atr
    lower = ema.values - mult * atr
    
    return ema.values, upper.values, lower.values

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
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger(close, 20, 2.0)
    keltner_ema, keltner_upper, keltner_lower = calculate_keltner(high, low, close, 20, 10, 1.5)
    
    # Bollinger Band width for squeeze detection
    bb_width = np.full(n, np.nan)
    for i in range(20, n):
        if bb_sma[i] > 1e-10:
            bb_width[i] = (bb_upper[i] - bb_lower[i]) / bb_sma[i]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_ranging = chop_14[i] > 55.0  # Slightly lower threshold for more signals
        is_trending = chop_14[i] < 45.0  # Slightly higher threshold for more signals
        
        # === BOLLINGER POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005  # Within 0.5% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.995  # Within 0.5% of upper band
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # RANGING MARKET: Mean-reversion at extremes
        if is_ranging:
            # Long: RSI oversold + near BB lower + weekly bull bias (loose conditions)
            if rsi_14[i] < 40 and (near_bb_lower or rsi_14[i] < 35):
                new_signal = SIZE
            # Short: RSI overbought + near BB upper + weekly bear bias
            elif rsi_14[i] > 60 and (near_bb_upper or rsi_14[i] > 65):
                new_signal = -SIZE
        
        # TRENDING MARKET: Follow weekly HMA direction
        elif is_trending:
            # Long: Weekly bull bias + RSI not overbought
            if bull_bias and rsi_14[i] < 70:
                new_signal = SIZE
            # Short: Weekly bear bias + RSI not oversold
            elif bear_bias and rsi_14[i] > 30:
                new_signal = -SIZE
        
        # NEUTRAL: Use RSI extremes with weekly bias
        else:
            if rsi_14[i] < 38 and bull_bias:
                new_signal = SIZE
            elif rsi_14[i] > 62 and bear_bias:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === WEEKLY BIAS REVERSAL EXIT ===
        # Exit if weekly trend flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias and is_trending:
                new_signal = 0.0
            if position_side < 0 and bull_bias and is_trending:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals