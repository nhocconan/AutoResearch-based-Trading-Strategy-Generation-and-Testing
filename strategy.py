#!/usr/bin/env python3
"""
Hypothesis: 15m primary with 4h HTF trend filter captures intraday moves while avoiding noise.
HMA(8/21) crossover for entry timing + Z-score(20) for overextension filter + volume confirmation.
4h HMA(21) determines primary trend direction - only trade in HTF trend direction.
ATR(14) stoploss at 2.5*ATR protects capital. SIZE=0.25 discrete levels minimize fee churn.
This differs from failed strategies by using HMA crossover + Z-score instead of Supertrend/Donchian.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_zscore_volume_15m_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response, smoother than EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean().values
    wma2 = close_s.ewm(span=period, adjust=False).mean().values
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_zscore(close, period):
    """Z-score for mean reversion detection"""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(period, min_periods=period).mean().values
    rolling_std = close_s.rolling(period, min_periods=period).std().values
    zscore = np.divide((close - rolling_mean), rolling_std, out=np.zeros_like(close), where=rolling_std>0)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    hma_4h = calculate_hma(close_4h, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # 15m indicators - all computed before loop (Rule 8)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # HMA(8) and HMA(21) for crossover signals
    hma8 = calculate_hma(close, 8)
    hma21 = calculate_hma(close, 21)
    
    # Z-score(20) for overextension detection
    zscore = calculate_zscore(close, 20)
    
    # ATR(14) for stoploss
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume ratio (current vs 20-bar average)
    volume_avg = volume_s.rolling(20, min_periods=20).mean().values
    volume_ratio = np.divide(volume, volume_avg, out=np.ones_like(volume), where=volume_avg>0)
    
    # EMA(50) for additional trend confirmation
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend: 4h HMA direction (Rule 2 - use aligned array)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # Local trend: HMA crossover
        hma_cross_long = hma8[i] > hma21[i] and hma8[i-1] <= hma21[i-1]
        hma_cross_short = hma8[i] < hma21[i] and hma8[i-1] >= hma21[i-1]
        
        # HMA alignment (already in trend)
        hma_aligned_long = hma8[i] > hma21[i] and close[i] > hma21[i]
        hma_aligned_short = hma8[i] < hma21[i] and close[i] < hma21[i]
        
        # Z-score filter (not overextended)
        zscore_ok_long = zscore[i] < 1.5  # not overbought
        zscore_ok_short = zscore[i] > -1.5  # not oversold
        
        # Z-score mean reversion entries
        zscore_oversold = zscore[i] < -2.0
        zscore_overbought = zscore[i] > 2.0
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] > 0.8  # at least 80% of avg volume
        
        # Price vs EMA50 filter
        above_ema50 = close[i] > ema50[i]
        below_ema50 = close[i] < ema50[i]
        
        # Stoploss and trailing logic (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trail_stop = highest_since_entry - 2.5 * atr[i]
            initial_stop = entry_price - 2.5 * atr[i]
            if close[i] < max(trail_stop, initial_stop):
                signals[i] = 0.0
                position_side = 0
                continue
        
        if position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trail_stop = lowest_since_entry + 2.5 * atr[i]
            initial_stop = entry_price + 2.5 * atr[i]
            if close[i] > min(trail_stop, initial_stop):
                signals[i] = 0.0
                position_side = 0
                continue
        
        # Entry logic - only enter when flat
        if position_side == 0:
            # Long: HTF bullish + HMA aligned + volume confirmed + not overextended
            if htf_bullish and hma_aligned_long and volume_confirmed and above_ema50:
                if zscore_ok_long:
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                # Mean reversion entry in uptrend (deep pullback)
                elif zscore_oversold and close[i] > hma_4h_aligned[i]:
                    signals[i] = SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
            
            # Short: HTF bearish + HMA aligned + volume confirmed + not overextended
            elif htf_bearish and hma_aligned_short and volume_confirmed and below_ema50:
                if zscore_ok_short:
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
                # Mean reversion entry in downtrend (sharp rally)
                elif zscore_overbought and close[i] < hma_4h_aligned[i]:
                    signals[i] = -SIZE
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
        else:
            # Hold position - maintain signal
            signals[i] = signals[i-1]
    
    return signals