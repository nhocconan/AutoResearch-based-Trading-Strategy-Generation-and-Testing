#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper band AND close > 1d EMA34 AND volume > 2.0x 12h volume average.
# Short when price breaks below Donchian lower band AND close < 1d EMA34 AND volume > 2.0x 12h volume average.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Volume spike filters low-momentum breakouts. 1d EMA34 ensures trades only in established daily trends.
# Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend).
# Target: 12-30 trades/year on 12h timeframe.

name = "12h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) on primary timeframe (12h)
    # Upper band: 20-period high
    # Lower band: 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume average (20-period) for volume spike filter
    volume_series = pd.Series(volume)
    vol_ma_12h = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Donchian, ATR, volume, and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_12h[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 2.0x 12h volume average
        if vol_ma_12h[i] <= 0 or np.isnan(vol_ma_12h[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_12h[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band AND close > 1d EMA34 AND volume spike
            if (curr_high > donchian_upper[i] and 
                curr_close > ema_34_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower band AND close < 1d EMA34 AND volume spike
            elif (curr_low < donchian_lower[i] and 
                  curr_close < ema_34_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower band (failed breakout)
            elif curr_close < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper band (failed breakdown)
            elif curr_close > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals