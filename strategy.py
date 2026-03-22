#!/usr/bin/env python3
"""
Experiment #498: 1d KAMA-Fisher Weekly Bias with Volume Confirmation

Hypothesis: After 497 failed experiments, the critical insight is that complex regime
filters (Choppiness, ADX, multiple conditions) are OVER-FILTERING and preventing
sufficient trades. The solution is SIMPLER logic with stronger individual signals:

1. KAMA (Kaufman Adaptive Moving Average) - adapts to market efficiency
   - Fast in trends (low friction), slow in ranges (high friction)
   - Better than EMA/HMA for crypto's regime changes
   - Crossover signals generate trades without excessive whipsaw

2. FISHER TRANSFORM (period=9) - normalized reversal indicator
   - Maps price to -1 to +1 range
   - Long when Fisher crosses above -0.8 (oversold reversal)
   - Short when Fisher crosses below +0.8 (overbought reversal)
   - More sensitive than RSI for daily timeframe

3. WEEKLY HMA(21) TREND BIAS (via mtf_data helper)
   - Only long when price > 1w HMA (bull bias)
   - Only short when price < 1w HMA (bear bias)
   - Prevents counter-trend trades that destroy Sharpe

4. VOLUME CONFIRMATION
   - Entry requires volume > 1.2 * 20-day avg volume
   - Filters false breakouts and weak reversals

5. ATR(14) TRAILING STOP at 2.5x
   - Tighter than 3.0x for daily timeframe
   - Protects capital during 2022-style crashes

6. POSITION SIZING: 0.25 discrete
   - Conservative for daily volatility
   - Discrete levels minimize fee churn

Why this should work on 1d:
- KAMA adapts to crypto's changing volatility regimes
- Fisher Transform catches reversals better than RSI
- Weekly HMA provides robust trend filter without whipsaw
- Volume confirmation ensures real moves, not noise
- Fewer filters = MORE TRADES (critical for Sharpe calculation)
- Should generate 25-50 trades/year per symbol

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_fisher_weekly_hma_volume_atr_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - fast in trends, slow in ranges.
    
    Efficiency Ratio (ER) = |Price Change| / Sum of Individual Price Changes
    Smoothing Constant (SC) = [ER * (fast - slow) + slow]^2
    KAMA = KAMA_prev + SC * (Price - KAMA_prev)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    price_change = np.abs(close - np.roll(close, period))
    price_change[:period] = np.nan
    
    sum_changes = np.full(n, np.nan)
    for i in range(period, n):
        sum_changes[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    er = price_change / sum_changes
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Calculate Fisher Transform.
    Normalizes price to -1 to +1 range for reversal detection.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X is normalized price
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    signal_line = np.full(n, np.nan)
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = close[i-period+1:i+1].max()
        lowest = close[i-period+1:i+1].min()
        
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        # Normalize price to -1 to +1
        x = (2.0 * (close[i] - lowest) / range_val) - 1.0
        x = np.clip(x, -0.99, 0.99)  # Prevent ln(0)
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Signal line (previous Fisher value)
        if i > period:
            signal_line[i] = fisher[i-1]
    
    return fisher, signal_line

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio relative to rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_avg.replace(0, np.inf)
    return vol_ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_regime = close[i] > hma_1w_aligned[i]
        bear_regime = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === KAMA TREND SIGNAL ===
        # KAMA crossover (current vs previous)
        kama_bullish = kama[i] > kama[i-1] if i > 0 else False
        kama_bearish = kama[i] < kama[i-1] if i > 0 else False
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNAL ===
        # Long: Fisher crosses above -0.8 from below (oversold reversal)
        fisher_long = fisher[i] > -0.8 and fisher_signal[i] <= -0.8
        
        # Short: Fisher crosses below +0.8 from above (overbought reversal)
        fisher_short = fisher[i] < 0.8 and fisher_signal[i] >= 0.8
        
        # === ENTRY LOGIC (simplified, fewer filters) ===
        new_signal = 0.0
        
        # LONG ENTRY: Bull regime + Fisher long + volume OR KAMA bullish
        if bull_regime:
            if fisher_long and volume_confirmed:
                new_signal = SIZE
            elif fisher_long and price_above_kama:
                new_signal = SIZE
            elif kama_bullish and price_above_kama and volume_confirmed:
                new_signal = SIZE
        
        # SHORT ENTRY: Bear regime + Fisher short + volume OR KAMA bearish
        if bear_regime:
            if fisher_short and volume_confirmed:
                new_signal = -SIZE
            elif fisher_short and price_below_kama:
                new_signal = -SIZE
            elif kama_bearish and price_below_kama and volume_confirmed:
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