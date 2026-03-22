#!/usr/bin/env python3
"""
Experiment #504: 1d Fisher Transform + Weekly HMA Trend with ADX Confirmation

Hypothesis: After analyzing 473+ failed experiments, the key insight is that daily 
timeframe strategies fail due to OVER-FILTERING. Complex regime filters (Chop + RSI + 
Z-score + ADX) create mutually exclusive conditions that rarely trigger. 

This strategy SIMPLIFIES while keeping the proven edges:

1. WEEKLY HMA(21) TREND BIAS (via mtf_data helper):
   - Long only when price > 1w HMA (bull regime)
   - Short only when price < 1w HMA (bear regime)
   - This single filter avoids 2022-style whipsaws

2. EHLERS FISHER TRANSFORM (period=9) FOR ENTRY TIMING:
   - Fisher is designed for non-normal distributions (crypto returns)
   - Long: Fisher crosses above -1.8 (oversold reversal)
   - Short: Fisher crosses below +1.8 (overbought reversal)
   - More responsive than RSI for daily timeframe

3. ADX(14) TREND CONFIRMATION WITH HYSTERESIS:
   - Enter when ADX > 20 (trending)
   - Stay in trade until ADX < 15 (trend weakening)
   - Hysteresis prevents rapid flip-flopping

4. ATR(14) TRAILING STOP at 2.5x:
   - Tighter than 3.0x for daily volatility
   - Signal → 0 when stoploss hit

5. POSITION SIZING: 0.28 discrete
   - Conservative for daily swings
   - Discrete levels (0.0, ±0.28) minimize fee churn

Why this should beat the baseline (Sharpe=0.676):
- Fisher Transform catches reversals better than RSI in bear markets
- Simpler logic = MORE TRADES (critical for Sharpe calculation)
- Weekly HMA bias prevents catastrophic 2022-style losses
- ADX hysteresis reduces whipsaw exits
- Should generate 20-40 trades/year per symbol

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_weekly_hma_adx_hysteresis_atr_v1"
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

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian-like distribution for better signal detection.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.66 * (price - min)/(max - min) - 0.66
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest = close[i-period+1:i+1].max()
        lowest = close[i-period+1:i+1].min()
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        x = 0.66 * ((close[i] - lowest) / price_range - 0.5) + 0.66 * 0.5
        x = 0.66 * ((close[i] - lowest) / price_range) - 0.33
        
        # Clamp to avoid division by zero
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Previous value for crossover detection
        if i > period - 1:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

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
    fisher, fisher_prev = calculate_fisher(close, 9)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss and ADX hysteresis
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_adx = 0.0
    
    for i in range(100, n):
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
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_regime = close[i] > hma_1w_aligned[i]
        bear_regime = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM CROSSOVER SIGNALS ===
        fisher_long_cross = (fisher_prev[i] < -1.8) and (fisher[i] >= -1.8)
        fisher_short_cross = (fisher_prev[i] > 1.8) and (fisher[i] <= 1.8)
        
        # === ADX TREND CONFIRMATION ===
        adx_trending = adx[i] > 20
        adx_weakening = adx[i] < 15
        
        # === ASYMMETRIC ENTRY LOGIC ===
        new_signal = 0.0
        
        # BULL REGIME: Only long entries
        if bull_regime:
            # Fisher oversold cross + ADX trending
            if fisher_long_cross and adx_trending:
                new_signal = SIZE
        
        # BEAR REGIME: Only short entries
        if bear_regime:
            # Fisher overbought cross + ADX trending
            if fisher_short_cross and adx_trending:
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
        
        # === ADX HYSTERESIS EXIT ===
        # Exit position when trend weakens (ADX < 15)
        if in_position and adx_weakening:
            new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if weekly trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                entry_adx = adx[i]
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                entry_adx = adx[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                entry_adx = 0.0
        
        signals[i] = new_signal
    
    return signals