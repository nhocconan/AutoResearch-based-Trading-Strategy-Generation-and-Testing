# 4h CCI + ATR Breakout with Volume and ADX Trend Filter
# Hypothesis: CCI identifies overbought/oversold conditions while ATR breakouts capture volatility expansion.
# Combined with ADX trend filter to avoid false signals in ranging markets and volume confirmation.
# Designed for low trade frequency (<25/year) to minimize fee decay while capturing strong moves.
# Works in both bull (breakouts up) and bear (breakouts down) markets via symmetric logic.

name = "4h_CCI_ATR_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === CCI (20) ===
    tp = (high + low + close) / 3.0
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(np.abs(tp - ma_tp)).rolling(window=20, min_periods=20).mean().values
    cci = (tp - ma_tp) / (0.015 * mad)
    
    # === ATR (14) for breakout and stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === ADX (14) for trend strength ===
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # === Donchian Channel (20) for breakout levels ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(cci[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + CCI > 100 (strong momentum) + ADX > 25 (trending) + volume spike
            if (close[i] > highest_high[i] and 
                cci[i] > 100 and
                adx[i] > 25 and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + CCI < -100 (strong momentum) + ADX > 25 (trending) + volume spike
            elif (close[i] < lowest_low[i] and 
                  cci[i] < -100 and
                  adx[i] > 25 and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR CCI < -100 (reversal) OR ADX < 20 (trend weakening)
            if close[i] < lowest_low[i] or cci[i] < -100 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR CCI > 100 (reversal) OR ADX < 20 (trend weakening)
            if close[i] > highest_high[i] or cci[i] > 100 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals