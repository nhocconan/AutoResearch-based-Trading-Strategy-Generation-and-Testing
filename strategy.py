#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 AND EMA34 rising AND volume > 1.5x 4h volume average.
# Short when price breaks below S3 AND EMA34 falling AND volume confirmation.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla levels calculated from prior completed 1d bar to avoid look-ahead.
# Volume spike filters low-momentum signals. 1d EMA34 ensures trades only in established daily trends.
# Works in bull (breakouts with uptrend) and bear (breakouts with downtrend).
# Target: 25-45 trades/year on 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels, EMA34, and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R4 = close + 1.1*(high-low)/2, R3 = close + 1.1*(high-low)/4, etc.
    # We only need R3 and S3
    rango = high_1d - low_1d
    r3 = close_1d + 1.1 * rango / 4
    s3 = close_1d - 1.1 * rango / 4
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d EMA34 slope (rising/falling)
    ema_slope = np.diff(ema_34_1d_aligned, prepend=ema_34_1d_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Calculate 4h volume average (20-period)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_rising[i]) if i < len(ema_rising) else True or 
            np.isnan(ema_falling[i]) if i < len(ema_falling) else True or 
            np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 1.5x 4h volume average
        if vol_ma_4h[i] <= 0 or np.isnan(vol_ma_4h[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_4h[i] * 1.5)
        
        # Breakout conditions
        breakout_above_r3 = curr_close > r3_aligned[i]
        breakout_below_s3 = curr_close < s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 AND EMA34 rising AND volume spike
            if (breakout_above_r3 and 
                ema_rising[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Breakout below S3 AND EMA34 falling AND volume spike
            elif (breakout_below_s3 and 
                  ema_falling[i] and 
                  volume_spike):
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
            # Exit: price crosses below R3 OR EMA34 turns falling
            elif (curr_close < r3_aligned[i]) or (not ema_rising[i]):
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
            # Exit: price crosses above S3 OR EMA34 turns rising
            elif (curr_close > s3_aligned[i]) or (not ema_falling[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals