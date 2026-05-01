#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold) AND price > 1d EMA50 AND volume > 1.5x 4h volume average.
# Short when Williams %R crosses below -20 (overbought) AND price < 1d EMA50 AND volume > 1.5x 4h volume average.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Williams %R calculated on completed 4h bar to avoid look-ahead.
# Volume spike filters low-momentum signals. 1d EMA50 ensures trades only in established trends.
# Works in bull (mean reversion in uptrend) and bear (mean reversion in downtrend).
# Target: 15-25 trades/year on 4h timeframe (60-100 total over 4 years).

name = "4h_WilliamsR_1dEMA50_Volume_v1"
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
    
    # Calculate Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Load 1d data ONCE before loop for EMA50 (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 4h data ONCE before loop for volume average (primary timeframe)
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
    
    # Start after warmup for Williams %R, EMA, ATR, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1]
        
        # Volume spike: current volume > 1.5x 4h volume average
        if vol_ma_4h_aligned[i] <= 0 or np.isnan(vol_ma_4h_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_4h_aligned[i] * 1.5)
        
        # Trend filter: price vs 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Williams %R signals: cross above -80 (long) or below -20 (short)
        williams_long_signal = (prev_williams_r <= -80) and (curr_williams_r > -80)
        williams_short_signal = (prev_williams_r >= -20) and (curr_williams_r < -20)
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND uptrend AND volume spike
            if (williams_long_signal and 
                uptrend and 
                volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R crosses below -20 AND downtrend AND volume spike
            elif (williams_short_signal and 
                  downtrend and 
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
            # Exit: Williams %R crosses below -50 (momentum loss) OR trend turns down
            elif (curr_williams_r < -50) or (not uptrend):
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
            # Exit: Williams %R crosses above -50 (momentum loss) OR trend turns up
            elif (curr_williams_r > -50) or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals