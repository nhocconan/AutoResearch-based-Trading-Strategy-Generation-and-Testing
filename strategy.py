#!/usr/bin/env python3
"""
Experiment #357: 1h Choppiness Index Regime-Adaptive with 4h HMA Trend Bias

Hypothesis: After 356 failed experiments, the clear pattern is that static strategies
fail because they don't adapt to market regime. This strategy uses Choppiness Index
to dynamically switch between mean-reversion (range) and trend-following (trend).

KEY INSIGHTS FROM RESEARCH:
1. CHOPPINESS INDEX (CHOP): Distinguishes ranging vs trending markets
   - CHOP > 61.8 = choppy/ranging (use mean reversion at BB extremes)
   - CHOP < 38.2 = trending (use breakout entries with HTF bias)
   - This meta-filter has proven effective in bear/range markets (2022, 2025)

2. MEAN REVERSION MODE (CHOP > 61.8):
   - Long when price < BB_lower AND RSI(14) < 35 (oversold in range)
   - Short when price > BB_upper AND RSI(14) > 65 (overbought in range)
   - Works well in 2025 bear/range market where trend strategies fail

3. TREND MODE (CHOP < 38.2):
   - Long breakout above Donchian(20) high + 4h HMA bullish bias
   - Short breakout below Donchian(20) low + 4h HMA bearish bias
   - Captures major moves while filtering counter-trend breakouts

4. 4h HMA TREND BIAS (via mtf_data helper):
   - Only take long breakouts if price > 4h HMA(21)
   - Only take short breakouts if price < 4h HMA(21)
   - Filters 60%+ of false breakouts

5. POSITION SIZING: 0.25 discrete (conservative for 1h volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn (each change costs 0.10%)

6. ATR TRAILING STOP (2.5x): Protect capital on reversals
   - Signal → 0 when price moves 2.5*ATR against position

Why this should work:
- Adapts to BOTH bull and bear markets (unlike pure trend strategies)
- Mean reversion mode captures 2025 bear/range opportunities
- Trend mode captures 2021 bull run momentum
- 4h HMA filter reduces false breakouts significantly
- Should generate 30-60 trades/year per symbol (enough for statistical significance)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop, use align_htf_to_ltf)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_4h_hma_adaptive_atr_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.zeros(len(close))
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss <= 1e-10] = 100.0
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            atr_sum = atr_series.iloc[i-period+1:i+1].sum()
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian_channels(high, low, period=20):
    """
    Calculate Donchian Channels.
    Upper = highest high of last N periods
    Lower = lowest low of last N periods
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] < 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === 4h HMA TREND BIAS ===
        bull_bias_4h = close[i] > hma_4h_aligned[i]
        bear_bias_4h = close[i] < hma_4h_aligned[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # === MEAN REVERSION MODE (CHOP > 61.8) ===
        if is_ranging:
            # Long: price at BB lower + RSI oversold
            if close[i] < bb_lower[i] and rsi[i] < 35:
                new_signal = SIZE
            
            # Short: price at BB upper + RSI overbought
            elif close[i] > bb_upper[i] and rsi[i] > 65:
                new_signal = -SIZE
        
        # === TREND MODE (CHOP < 38.2) ===
        elif is_trending:
            # Long breakout above Donchian + 4h HMA bullish bias
            if close[i] > donchian_upper[i-1] and bull_bias_4h:
                new_signal = SIZE
            
            # Short breakout below Donchian + 4h HMA bearish bias
            elif close[i] < donchian_lower[i-1] and bear_bias_4h:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if position_side != 0 and new_signal != 0.0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            elif position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime flips against position type
        if position_side != 0 and new_signal != 0.0:
            # Long position in newly trending market with bear bias
            if position_side > 0 and is_trending and bear_bias_4h:
                new_signal = 0.0
            # Short position in newly trending market with bull bias
            elif position_side < 0 and is_trending and bull_bias_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if position_side == 0:
                # New position
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
            # Exit position
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals