#!/usr/bin/env python3
"""
Experiment #024: 1d RSI Extreme + 1w Trend + Volume + Choppiness Regime

HYPOTHESIS: 1d RSI(14) extremes (<25 for long, >75 for short) provide
reliable mean-reversion signals. Combined with:
- 1w EMA21 for trend direction (filters counter-trend trades)
- Volume confirmation (1.3x 20d avg)
- Choppiness filter (skip when CHOP > 60)

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: RSI < 25 (oversold) + price > 1w EMA21 + volume spike = buy the dip
- Bear: RSI > 75 (overbought) + price < 1w EMA21 + volume spike = short the rally
- Range (CHOP > 60): Skip entries entirely, avoid whipsaws
- Trending (CHOP < 50): Allow mean-reversion entries aligned with trend

EXPECTED TRADES: 30-60 total over 4 years (7-15/year)
- RSI extremes: ~40-80 triggers/year (30-60 per direction)
- Volume filter (1.3x): ~50% pass
- Choppiness filter (CHOP < 60): ~40% pass
- 1w EMA21 trend: ~50% pass
- Final: ~8-15/year = 32-60 total over 4 years → statistical validity
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_extreme_1w_trend_vol_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    close = prices["close"].values if isinstance(prices, pd.DataFrame) else prices
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_losses = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_losses != 0, avg_gains / avg_losses, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures market choppiness vs trending
    CHOP > 61.8 = choppy (range-bound), CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            atr_sum = np.sum([
                max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                for j in range(i-period+1, i+1)
            ])
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA21 for trend direction
    ema21_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # === Local 1d indicators ===
    # RSI(14)
    rsi_14 = calculate_rsi(prices, period=14)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume average (20 bars = ~20 trading days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 60  # Enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(rsi_14[i]) or np.isnan(atr_14[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === Regime check ===
        is_not_choppy = chop[i] < 60.0  # Below 60 = not choppy enough to skip
        
        # === 1w trend direction ===
        # Need at least 2 valid values for trend direction
        if np.isnan(ema21_1w_aligned[i]) or np.isnan(ema21_1w_aligned[i-1]):
            htf_bull = close[i] > close[i-1]  # Fallback to local trend
            htf_bear = close[i] < close[i-1]
        else:
            htf_bull = ema21_1w_aligned[i] > ema21_1w_aligned[i-1]
            htf_bear = ema21_1w_aligned[i] < ema21_1w_aligned[i-1]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Volume spike confirmation
        vol_spike = vol_ratio[i] > 1.3
        
        # RSI extremes
        rsi_oversold = rsi_14[i] < 25.0
        rsi_overbought = rsi_14[i] > 75.0
        
        # === LONG ENTRY: RSI oversold + 1w trend up + volume spike ===
        if not in_position:
            if rsi_oversold and htf_bull and vol_spike and is_not_choppy:
                desired_signal = SIZE
                
            # === SHORT ENTRY: RSI overbought + 1w trend down + volume spike ===
            elif rsi_overbought and htf_bear and vol_spike and is_not_choppy:
                desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Long: stop if price drops 2.5 ATR from entry
                stop_distance = entry_price - 2.5 * entry_atr
                if low[i] < stop_distance:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # Exit if RSI returns to neutral (>55, not overbought)
                elif rsi_14[i] > 55:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Short: stop if price rises 2.5 ATR from entry
                stop_distance = entry_price + 2.5 * entry_atr
                if high[i] > stop_distance:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # Exit if RSI returns to neutral (<45, not oversold)
                elif rsi_14[i] < 45:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 3 days to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        
        signals[i] = desired_signal
    
    return signals