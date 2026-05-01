#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 with 12h EMA34 uptrend and volume > 2.0x 24-bar average.
# Short when price breaks below Camarilla S3 with 12h EMA34 downtrend and volume confirmation.
# Uses discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Primary timeframe: 6h, HTF: 12h for EMA trend filter and Camarilla levels.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.
# Session filter: 08-20 UTC to reduce noise trades.

name = "6h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for EMA trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA34 trend filter
    ema_34 = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Calculate ATR(14) for stoploss from 12h data
    tr1 = df_12h['high'].values[1:] - df_12h['low'].values[1:]
    tr2 = np.abs(df_12h['high'].values[1:] - df_12h['close'].values[:-1])
    tr3 = np.abs(df_12h['low'].values[1:] - df_12h['close'].values[:-1])
    tr_12h = np.concatenate([[np.max([df_12h['high'].values[0] - df_12h['low'].values[0], 
                                       np.abs(df_12h['high'].values[0] - df_12h['close'].values[0]), 
                                       np.abs(df_12h['low'].values[0] - df_12h['close'].values[0])])], 
                              np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate Camarilla levels from 12h data (based on previous day's OHLC)
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    #                   S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = close, H = high, L = low of previous 12h bar
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Shift by 1 to use previous bar's OHLC for current bar's levels
    prev_close = np.roll(close_12h, 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close[0] = np.nan  # first bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 50  # warmup for EMA34 and ATR
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 24-bar average (tight to reduce trades)
        vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 2.0)
        
        # Camarilla breakout conditions
        breakout_long = curr_high > camarilla_r3_aligned[i]  # price breaks above R3
        breakout_short = curr_low < camarilla_s3_aligned[i]  # price breaks below S3
        
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
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below S3 OR trend turns bearish
            elif (curr_low < camarilla_s3_aligned[i] or 
                  bearish_trend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above R3 OR trend turns bullish
            elif (curr_high > camarilla_r3_aligned[i] or 
                  bullish_trend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals