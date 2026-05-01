#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume spike + ADX trend filter + ATR stoploss.
# Long when price breaks above Donchian(20) high AND volume > 2.0x 4h volume average AND ADX(14) > 25.
# Short when price breaks below Donchian(20) low AND volume > 2.0x 4h volume average AND ADX(14) > 25.
# Uses discrete sizing 0.30. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Works in bull (breakouts with volume in uptrend) and bear (breakdowns with volume in downtrend).
# Target: 25-40 trades/year on 4h timeframe.

name = "4h_Donchian20_VolumeSpike_ADX_Trend_v1"
timeframe = "4h"
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
    
    # Calculate ADX(14) for trend filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 4h data ONCE before loop for volume filters (primary timeframe data)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, ADX, Donchian, volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 2.0x 4h volume average
        if vol_ma_4h_aligned[i] <= 0 or np.isnan(vol_ma_4h_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_4h_aligned[i] * 2.0)
        
        # Donchian breakout
        breakout_high = curr_high > donchian_high[i]
        breakout_low = curr_low < donchian_low[i]
        
        # ADX trend filter
        strong_trend = adx[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian high AND volume spike AND strong trend
            if breakout_high and volume_spike and strong_trend:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short: Breakout below Donchian low AND volume spike AND strong trend
            elif breakout_low and volume_spike and strong_trend:
                signals[i] = -0.30
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
            # Exit: price closes below Donchian middle (mean reversion)
            elif curr_close < (donchian_high[i] + donchian_low[i]) / 2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price closes above Donchian middle (mean reversion)
            elif curr_close > (donchian_high[i] + donchian_low[i]) / 2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
    
    return signals