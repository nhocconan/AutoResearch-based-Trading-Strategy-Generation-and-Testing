#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion + 1w EMA200 trend filter + volume confirmation.
# Long when Williams %R < -80 (oversold) AND price > 1w EMA200 (bullish bias) AND volume > 1.5x 20-day average.
# Short when Williams %R > -20 (overbought) AND price < 1w EMA200 (bearish bias) AND volume > 1.5x 20-day average.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Williams %R identifies exhaustion points in ranging/bear markets; weekly trend filter avoids counter-trend trades.
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years).

name = "1d_WilliamsR_Overextended_1wEMA200_Volume_v1"
timeframe = "1d"
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
    
    # Calculate 1d Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0,
                          -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14),
                          -50)  # neutral when range is zero
    
    # Calculate 1w EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 20-day volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Williams %R, EMA, and volume
    start_idx = 200  # EMA200 requires 200 periods
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        if vol_ma_20[i] <= 0 or np.isnan(vol_ma_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_20[i] * 1.5)
        
        # Trend filter: price vs 1w EMA200
        uptrend_bias = curr_close > ema_200_1w_aligned[i]
        downtrend_bias = curr_close < ema_200_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND uptrend bias AND volume confirmation
            if (williams_r[i] < -80 and 
                uptrend_bias and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R > -20 (overbought) AND downtrend bias AND volume confirmation
            elif (williams_r[i] > -20 and 
                  downtrend_bias and 
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
            # Exit: Williams %R > -50 (exiting oversold) OR trend bias turns down
            elif (williams_r[i] > -50) or (not uptrend_bias):
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
            # Exit: Williams %R < -50 (exiting overbought) OR trend bias turns up
            elif (williams_r[i] < -50) or (not downtrend_bias):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals