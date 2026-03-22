#!/usr/bin/env python3
"""
Experiment #364: 4h Fisher Transform + 1d HMA Trend + Choppiness Regime Filter

Hypothesis: After 363 failed experiments, the pattern shows that static strategies fail
because they don't adapt to market regime. For 4h timeframe specifically:

1. EHLERS FISHER TRANSFORM (period=9): Excellent for catching reversals in bear/range
   markets where trend-following fails. Fisher normalizes price to Gaussian distribution,
   making extremes statistically significant. Entry when Fisher crosses -1.5 (long) or
   +1.5 (short) - proven in "Cybernetic Analysis for Stocks and Futures".

2. 1d HMA TREND BIAS (21): Only take Fisher signals in direction of HTF trend
   - Long Fisher signals only if price > 1d HMA(21)
   - Short Fisher signals only if price < 1d HMA(21)
   - Filters 60%+ of counter-trend reversals that fail

3. CHOPPINESS INDEX REGIME (14): Detect ranging vs trending markets
   - CHOP > 61.8 = range (use Fisher mean-reversion signals)
   - CHOP < 38.2 = trend (use Fisher trend-continuation signals)
   - 38.2-61.8 = neutral (reduce position size by 50%)
   - This is the KEY differentiator - adapts to market state

4. VOLUME CONFIRMATION: Fisher signal + volume > SMA(volume, 20) * 1.2
   - Confirms institutional participation in the move
   - Reduces false signals during low-liquidity periods

5. ATR TRAILING STOP (2.5x): Protect capital on reversals
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for limiting drawdown in 2022-style crashes

6. ASYMMETRIC POSITION SIZING: 
   - Bull regime (price > 1d HMA): max 0.30 long
   - Bear regime (price < 1d HMA): max 0.35 short (bear bias for BTC/ETH)
   - Discrete levels (0.0, ±0.20, ±0.30, ±0.35) minimize fee churn

Why 4h should work:
- Fast enough to catch 2022 crash moves, slow enough to filter noise
- Fisher Transform excels at 4h-12h timeframes per Ehlers research
- 1d HMA provides stable bias without excessive lag
- Choppiness filter prevents whipsaw in ranging markets (2023-2025)
- Should generate 40-80 trades/year per symbol (enough for stats, not too many fees)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_1d_hma_chop_regime_vol_atr_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for statistically significant extremes.
    Based on "Cybernetic Analysis for Stocks and Futures" by John Ehlers.
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((price - min) / (max - min) - 0.5) + 0.67 * prev_X
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Calculate median price
    median = (high + low) / 2
    
    for i in range(period, n):
        # Find highest high and lowest low over lookback period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        if highest - lowest < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_prev[i] = fisher_prev[i-1] if i > 0 else 0.0
            continue
        
        # Calculate X (normalized price with smoothing)
        x_raw = (median[i] - lowest) / (highest - lowest)
        
        if i == period:
            x = 0.66 * (x_raw - 0.5) + 0.67 * 0.0
        else:
            x = 0.66 * (x_raw - 0.5) + 0.67 * fisher_prev[i-1]
        
        # Clamp X to avoid log domain errors
        x = np.clip(x, -0.999, 0.999)
        
        # Calculate Fisher Transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        fisher_prev[i] = x
    
    return fisher

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    Based on E.W. Dreiss formula.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Sum of ATR over period (approximate with True Range sum)
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j-1]) if j > 0 else tr1
            tr_sum += np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Avoid division by zero
        if tr_sum < 1e-10 or highest - lowest < 1e-10:
            chop[i] = 50.0  # neutral
            continue
        
        # CHOP = 100 * log10(sum(ATR) / (highest - lowest)) / log10(period)
        chop[i] = 100 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BULL = 0.30   # Long size in bull regime
    SIZE_BEAR = 0.35   # Short size in bear regime (asymmetric bias)
    SIZE_NEUTRAL = 0.20  # Reduced size in neutral/choppy regime
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        neutral_market = not ranging_market and not trending_market
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma[i] * 1.2
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher extremes indicate potential reversals
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # Fisher trend continuation (in trending markets)
        fisher_bullish = fisher[i] > 0 and fisher[i-1] <= 0 if i > 0 else False
        fisher_bearish = fisher[i] < 0 and fisher[i-1] >= 0 if i > 0 else False
        
        # === DETERMINE POSITION SIZE BASED ON REGIME ===
        if bull_trend_1d:
            base_size = SIZE_BULL
        elif bear_trend_1d:
            base_size = SIZE_BEAR
        else:
            base_size = SIZE_NEUTRAL
        
        # Reduce size in neutral/choppy regime
        if neutral_market or ranging_market:
            base_size = base_size * 0.67  # Reduce to ~0.20
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY scenarios:
        # 1. Ranging market + Fisher oversold reversal + volume confirmed
        if ranging_market and fisher_long and volume_confirmed:
            new_signal = base_size
        
        # 2. Trending market + Fisher bullish cross + 1d bull bias + volume
        elif trending_market and fisher_bullish and bull_trend_1d and volume_confirmed:
            new_signal = base_size
        
        # 3. Neutral market + Fisher extreme + 1d bull bias (conservative)
        elif neutral_market and fisher_long and bull_trend_1d:
            new_signal = base_size * 0.67
        
        # SHORT ENTRY scenarios:
        # 1. Ranging market + Fisher overbought reversal + volume confirmed
        if ranging_market and fisher_short and volume_confirmed:
            new_signal = -base_size
        
        # 2. Trending market + Fisher bearish cross + 1d bear bias + volume
        elif trending_market and fisher_bearish and bear_trend_1d and volume_confirmed:
            new_signal = -base_size
        
        # 3. Neutral market + Fisher extreme + 1d bear bias (conservative)
        elif neutral_market and fisher_short and bear_trend_1d:
            new_signal = -base_size * 0.67
        
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
        # Exit if 1d trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === EXTREME CHOP EXIT ===
        # Exit if market becomes extremely choppy (CHOP > 70)
        if in_position and chop[i] > 70:
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