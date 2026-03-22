#!/usr/bin/env python3
"""
Experiment #442: 4h Fisher Transform + Choppiness Regime with Daily/Weekly HMA Filter

Hypothesis: After 441 experiments, the key insight is that BTC/ETH need REGIME-ADAPTIVE
strategies that work in both bull (2021) and bear (2022, 2025) markets. Simple trend
following fails because 2022 crash (-77%) and 2025 bear market destroy gains.

This strategy combines:
1. DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 1d HMA
   - Short bias when price < 1d HMA
   - HMA is smoother than EMA with less lag (Hull formula)

2. WEEKLY HMA(21) META-FILTER (via mtf_data helper):
   - Only take longs when price > 1w HMA (major bull trend)
   - Only take shorts when price < 1w HMA (major bear trend)
   - Prevents counter-trend disasters in major reversals

3. CHOPPINESS INDEX (14) REGIME DETECTION:
   - CHOP > 61.8 = ranging market (use mean reversion entries)
   - CHOP < 38.2 = trending market (use breakout entries)
   - 38.2-61.8 = neutral (require stronger signals)
   - This is the KEY differentiator from failed strategies

4. EHLERS FISHER TRANSFORM (period=9) FOR ENTRY TIMING:
   - Long: Fisher crosses above -1.5 (oversold reversal)
   - Short: Fisher crosses below +1.5 (overbought reversal)
   - Fisher normalizes price to Gaussian distribution, catches extremes
   - Proven edge in bear market rallies (2022 bottom, 2025 range)

5. VOLUME CONFIRMATION:
   - Entry requires volume > SMA(volume, 20) * 0.8
   - Prevents false breakouts on low volume

6. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crash protection

7. POSITION SIZING: 0.30 discrete (conservative for 4h volatility)
   - Max 30% capital per position
   - Discrete levels minimize fee churn

Why this should beat current best (Sharpe=0.676):
- Fisher Transform catches reversals better than RSI (proven in literature)
- Choppiness Index prevents trend-follow whipsaws in ranges
- Dual HTF filter (1d + 1w) prevents major counter-trend losses
- Volume confirmation reduces false signals
- Should generate 20-40 trades/year per symbol (sufficient frequency)
- Works on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_regime_1d_1w_hma_vol_atr_v1"
timeframe = "4h"
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
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extremes better than RSI.
    Reference: Ehlers, J.F. (2002) "Fisher Transform"
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_hl2 = high[i-period+1:i+1].max()
        lowest_hl2 = low[i-period+1:i+1].min()
        
        # Normalize to 0-1 range with bounds
        range_hl = highest_hl2 - lowest_hl2
        if range_hl < 1e-10:
            continue
        
        normalized = (hl2 - lowest_hl2) / range_hl
        
        # Bound to prevent division issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with previous value
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
        else:
            fisher[i] = fisher_val
        
        # Trigger line (1-period lag)
        if i > period:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    Reference: E.W. Dreiss
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        # Sum of ATR over period (approximated as sum of true ranges)
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr_sum += max(tr1, tr2, tr3)
        
        # Choppiness formula
        range_hl = highest_high - lowest_low
        if range_hl > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

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

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
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
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY HMA META-FILTER ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        neutral_market = 38.2 <= chop[i] <= 61.8
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_cross_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # Fisher extreme levels (for ranging market)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > vol_sma[i] * 0.8
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # TRENDING MARKET: Use Fisher crosses with trend alignment
        if trending_market:
            if fisher_cross_long and bull_trend_1d and bull_trend_1w and volume_ok:
                new_signal = SIZE
            elif fisher_cross_short and bear_trend_1d and bear_trend_1w and volume_ok:
                new_signal = -SIZE
        
        # RANGING MARKET: Use Fisher extremes (mean reversion)
        elif ranging_market:
            if fisher_oversold and bull_trend_1d and volume_ok:
                new_signal = SIZE
            elif fisher_overbought and bear_trend_1d and volume_ok:
                new_signal = -SIZE
        
        # NEUTRAL MARKET: Require stronger confirmation
        elif neutral_market:
            # Require both daily and weekly trend alignment
            if fisher_cross_long and bull_trend_1d and bull_trend_1w and volume_ok:
                new_signal = SIZE
            elif fisher_cross_short and bear_trend_1d and bear_trend_1w and volume_ok:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
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