#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 8-period SMA AND 1d EMA50 > EMA200 AND volume > 1.5x 20-bar average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 8-period SMA AND 1d EMA50 < EMA200 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Alligator based on SMAs of median price (hlc3): jaws=13, teeth=8, lips=5.
# Williams Alligator identifies trend phases; 1d EMA filter ensures alignment with higher timeframe trend.
# Volume confirmation filters weak breakouts. ATR stoploss manages risk.
# Works in bull (trend continuation) and bear (trend continuation) regimes by following Alligator alignment.

name = "12h_WilliamsAlligator_1dEMA_Volume_v1"
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
    
    # Calculate Williams Alligator on 12h timeframe
    # Median price = (high + low + close) / 3
    median_price = (high + low + close) / 3.0
    
    # Alligator lines: jaws (13-period SMA, 8-bar shift), teeth (8-period SMA, 5-bar shift), lips (5-period SMA, 3-bar shift)
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 8-period SMA for entry filter
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50 and EMA200
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMAs to 12h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Alligator (max shift 8 + jaws period 13 = 21) and ATR
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(sma_8[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0 or np.isnan(vol_ma):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.5)
        
        # Williams Alligator conditions
        bullish_alignment = jaws[i] < teeth[i] and teeth[i] < lips[i]
        bearish_alignment = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        # 1d trend filter: EMA50 > EMA200 for bullish, EMA50 < EMA200 for bearish
        uptrend_1d = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        downtrend_1d = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment AND price > SMA8 AND 1d uptrend AND volume confirmation
            if (bullish_alignment and 
                curr_close > sma_8[i] and 
                uptrend_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish Alligator alignment AND price < SMA8 AND 1d downtrend AND volume confirmation
            elif (bearish_alignment and 
                  curr_close < sma_8[i] and 
                  downtrend_1d and 
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
            # Exit: Alligator alignment turns bearish OR price crosses below SMA8
            elif not bullish_alignment or curr_close < sma_8[i]:
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
            # Exit: Alligator alignment turns bullish OR price crosses above SMA8
            elif not bearish_alignment or curr_close > sma_8[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals