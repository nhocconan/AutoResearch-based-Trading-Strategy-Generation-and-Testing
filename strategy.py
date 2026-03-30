#!/usr/bin/env python3
"""
Experiment #025: 4h Fisher Transform Momentum + Volume Spike + ADX Regime

HYPOTHESIS: Fisher Transform (period=9) provides excellent reversal signals by
normalizing price into Gaussian distribution. Combined with ADX regime filter
(ADX<25 = ranging, avoid), volume spike confirmation, and HTF trend alignment,
this captures mean-reversion opportunities in both bull and bear markets.

WHY IT WORKS IN BULL + BEAR:
- Bull: Fisher crosses above -1.5 + volume spike + HTF EMA200 confirms = buy dip
- Bear: Fisher crosses below +1.5 + volume spike + HTF EMA200 confirms = short rally
- Range: ADX<20 = no trades (avoids whipsaws)
- ADX>25 = trending (trade with momentum)

KEY INSIGHT FROM DB: Strategies with ~75-150 train trades succeed. Overtrading
(400+ trades) is the #1 killer. This strategy uses VERY strict entry conditions:
Fisher crossover + volume spike + ADX regime + HTF trend = 4 filters = few trades.

TARGET: 50-150 total trades over 4 years. Signal: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_volume_adx_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher(high, low, period=9):
    """
    Fisher Transform: transforms price into Gaussian distribution.
    Values > 2.0 = overbought (reversal likely)
    Values < -2.0 = oversold (reversal likely)
    Signal lines help identify crossover points
    """
    n = len(close)
    if n < period:
        return np.full(n, 0.0), np.full(n, 0.0)
    
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    fisher = np.zeros(n)
    signal = np.zeros(n)
    
    for i in range(period, n):
        hl2 = max_high[i] - min_low[i]
        if hl2 > 1e-10:
            value = 0.5 * (2 * (high[i] - min_low[i]) / hl2 - 1)
            
            # Smooth with EMA
            if i == period:
                fish = value
            else:
                fish = 0.6 * fisher[i-1] + 0.4 * value
            
            fisher[i] = np.clip(fish * 3.0, -5.0, 5.0)
            signal[i] = fisher[i-1]  # Trigger line
    
    return fisher, signal

def calculate_adx(high, low, close, period=14):
    """ADX - Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.maximum(atr, 1e-10))
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.maximum(atr, 1e-10))
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for multi-timeframe trend (faster than EMA200 for signals)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Local 4h indicators ===
    fisher, fisher_signal = calculate_fisher(high, low, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Fisher period * 2
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === FISHER CROSSOVER SIGNALS ===
        # Long: Fisher crosses above signal (was below -1.0)
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher[i-1] <= fisher_signal[i-1]
        # Short: Fisher crosses below signal (was above +1.0)
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher[i-1] >= fisher_signal[i-1]
        
        # === FISHER EXTREME LEVELS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === ADX REGIME FILTER ===
        adx_val = adx[i]
        adx_trending = adx_val > 25 if not np.isnan(adx_val) else False
        adx_weak = adx_val < 20 if not np.isnan(adx_val) else False
        
        # === HTF TREND ===
        above_htf_ema = close[i] > ema_50_aligned[i]
        below_htf_ema = close[i] < ema_50_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DI DIRECTION ===
        plus_di_val = plus_di[i] if not np.isnan(plus_di[i]) else 50
        minus_di_val = minus_di[i] if not np.isnan(minus_di[i]) else 50
        di_bullish = plus_di_val > minus_di_val
        di_bearish = minus_di_val > plus_di_val
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === STRICT LONG ENTRY ===
            # Fisher crossover UP + extreme oversold + volume spike + HTF uptrend + DI bullish
            # OR: Fisher extreme + bounce confirmation + volume
            if (fisher_cross_up and fisher_oversold) or (fisher[i] < -1.0 and close[i] > close[i-1]):
                # Volume confirmation required
                if vol_spike:
                    # Check HTF trend alignment
                    if above_htf_ema or adx_weak:  # Allow in range but prefer trend
                        # DI should support or neutral
                        if di_bullish or adx_weak:
                            desired_signal = SIZE
            
            # === STRICT SHORT ENTRY ===
            # Fisher crossover DOWN + extreme overbought + volume spike + HTF downtrend + DI bearish
            if (fisher_cross_down and fisher_overbought) or (fisher[i] > 1.0 and close[i] < close[i-1]):
                # Volume confirmation required
                if vol_spike:
                    # Check HTF trend alignment
                    if below_htf_ema or adx_weak:  # Allow in range but prefer trend
                        # DI should support or neutral
                        if di_bearish or adx_weak:
                            desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position:
            bars_held = i - entry_bar
            
            if position_side > 0:
                # Long stop: trail based on recent lows
                lowest_low = np.min(low[max(entry_bar, i-5):i+1]) if i > entry_bar else low[entry_bar]
                stop_price = lowest_low - 1.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                # Exit if Fisher flips bearish with confirmation
                elif fisher[i] < -1.5 and fisher_signal[i] < fisher_signal[i-1]:
                    desired_signal = 0.0
                # Take profit at 2R
                elif close[i] > entry_price + 2.5 * entry_atr:
                    desired_signal = SIZE / 2  # Reduce position
                    # Trail stop after
                    if close[i] > entry_price + 3 * entry_atr:
                        new_stop = close[i] - 1.5 * atr_14[i]
                        if new_stop > stop_price:
                            stop_price = new_stop
            
            elif position_side < 0:
                # Short stop: trail based on recent highs
                highest_high = np.max(high[max(entry_bar, i-5):i+1]) if i > entry_bar else high[entry_bar]
                stop_price = highest_high + 1.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                # Exit if Fisher flips bullish with confirmation
                elif fisher[i] > 1.5 and fisher_signal[i] > fisher_signal[i-1]:
                    desired_signal = 0.0
                # Take profit at 2R
                elif close[i] < entry_price - 2.5 * entry_atr:
                    desired_signal = -SIZE / 2  # Reduce position
                    # Trail stop after
                    if close[i] < entry_price - 3 * entry_atr:
                        new_stop = close[i] + 1.5 * atr_14[i]
                        if new_stop < stop_price:
                            stop_price = new_stop
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals