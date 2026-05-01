#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
# Long when Alligator jaws (13-period SMA) > teeth (8-period SMA) > lips (5-period SMA) AND close > 1d EMA34 AND volume > 1.5x 20-period volume median.
# Short when Alligator jaws < teeth < lips AND close < 1d EMA34 AND volume > 1.5x 20-period volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Alligator identifies trend alignment; 1d EMA34 filters for higher-timeframe trend; volume confirms conviction.
# Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years) to minimize fee drag.

name = "12h_WilliamsAlligator_1dEMA34_Volume_v1"
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
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Williams Alligator (5,8,13 SMAs) on primary timeframe
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # 13-period SMA (jaws)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # 8-period SMA (teeth)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # 5-period SMA (lips)
    
    # Calculate 1d EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Alligator, EMA, volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Alligator trend filter: jaws > teeth > lips (uptrend) or jaws < teeth < lips (downtrend)
        uptrend = jaw[i] > teeth[i] and teeth[i] > lips[i]
        downtrend = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend AND price > 1d EMA34 AND volume spike
            if uptrend and curr_close > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Alligator downtrend AND price < 1d EMA34 AND volume spike
            elif downtrend and curr_close < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator trend turns down OR price < 1d EMA34
            elif not uptrend or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator trend turns up OR price > 1d EMA34
            elif not downtrend or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals