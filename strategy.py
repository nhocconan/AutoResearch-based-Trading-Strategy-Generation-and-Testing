#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level with volume > 1.5x 20-period volume average and price > 1d EMA34.
# Short when price breaks below Camarilla S3 level with volume confirmation and price < 1d EMA34.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Camarilla levels calculated from prior completed 1d bar to avoid look-ahead.
# Volume confirmation filters low-momentum breakouts. 1d EMA34 ensures trades only in established trends.
# Works in bull (breakouts with strong uptrend) and bear (breakouts with strong downtrend) regimes.
# Target: 25-35 trades/year on 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Load 1d data ONCE before loop for EMA, volume, and Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Camarilla levels (R3, S3) from prior completed 1d bar
    # Camarilla formulas:
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    camarilla_r3 = close_1d_arr + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d_arr - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and Camarilla
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_1d_aligned[i] * 1.5)
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla R3 breakout up AND volume confirmation AND uptrend
            if (curr_high > camarilla_r3_aligned[i] and 
                volume_confirm and 
                uptrend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Camarilla S3 breakout down AND volume confirmation AND downtrend
            elif (curr_low < camarilla_s3_aligned[i] and 
                  volume_confirm and 
                  downtrend):
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
            # Exit: price re-enters Camarilla range (between S3 and R3) OR trend reverses
            elif (curr_low >= camarilla_s3_aligned[i] and curr_low <= camarilla_r3_aligned[i]) or \
                 (curr_close < ema_34_1d_aligned[i]):  # trend reversal
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
            # Exit: price re-enters Camarilla range (between S3 and R3) OR trend reverses
            elif (curr_high >= camarilla_s3_aligned[i] and curr_high <= camarilla_r3_aligned[i]) or \
                 (curr_close > ema_34_1d_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals