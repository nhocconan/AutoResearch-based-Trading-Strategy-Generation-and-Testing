#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above R3 with 1d EMA34 uptrend and volume > 2.0x 20-bar average.
# Short when price breaks below S3 with 1d EMA34 downtrend and volume confirmation.
# Uses discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR).
# Primary timeframe: 4h, HTF: 1d for EMA trend filter.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
# Camarilla levels from 1d provide strong intraday support/resistance with high win rate in ranging markets.

name = "4h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar (need 1d OHLC)
    # We'll use the 1d data to compute Camarilla for the current 4h bar
    # Camarilla formula based on previous day's range
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_R3 = np.zeros(len(df_1d))
    camarilla_S3 = np.zeros(len(df_1d))
    camarilla_R4 = np.zeros(len(df_1d))  # for exit
    camarilla_S4 = np.zeros(len(df_1d))  # for exit
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_R3[i] = np.nan
            camarilla_S3[i] = np.nan
            camarilla_R4[i] = np.nan
            camarilla_S4[i] = np.nan
        else:
            close_prev = df_1d_close[i-1]
            high_prev = df_1d_high[i-1]
            low_prev = df_1d_low[i-1]
            range_prev = high_prev - low_prev
            camarilla_R3[i] = close_prev + range_prev * 1.1 / 4
            camarilla_S3[i] = close_prev - range_prev * 1.1 / 4
            camarilla_R4[i] = close_prev + range_prev * 1.1 / 2
            camarilla_S4[i] = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 50  # warmup for EMA34 and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-bar average (tighter for fewer trades)
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        # Camarilla breakout conditions
        breakout_long = curr_high > camarilla_R3_aligned[i]  # price breaks above R3
        breakout_short = curr_low < camarilla_S3_aligned[i]  # price breaks below S3
        
        # Trend filter: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = curr_close > ema_34_aligned[i]
        bearish_trend = curr_close < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 AND bullish trend AND volume confirmation
            if (breakout_long and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Breakout below S3 AND bearish trend AND volume confirmation
            elif (breakout_short and 
                  bearish_trend and 
                  volume_confirm):
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
            # Exit: price breaks below S3 OR trend turns bearish OR hits R4 (profit target)
            elif (curr_low < camarilla_S3_aligned[i] or 
                  bearish_trend or
                  curr_high > camarilla_R4_aligned[i]):
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
            # Exit: price breaks above R3 OR trend turns bullish OR hits S4 (profit target)
            elif (curr_high > camarilla_R3_aligned[i] or 
                  bullish_trend or
                  curr_low < camarilla_S4_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals