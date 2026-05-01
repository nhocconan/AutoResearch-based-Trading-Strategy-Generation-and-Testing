#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA34 trend filter and volume confirmation.
# Long when price > Alligator Jaw (blue line) AND Teeth > Lips (bullish alignment) AND close > 1w EMA34 AND volume > 1.5x 20-period volume median.
# Short when price < Alligator Jaw AND Teeth < Lips (bearish alignment) AND close < 1w EMA34 AND volume > 1.5x 20-period volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Williams Alligator identifies trend phases via smoothed medians; 1w EMA34 filters for long-term trend alignment.
# Volume confirmation ensures breakout conviction. Works in bull markets (teeth above lips) and bear markets (teeth below lips).
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years).

name = "1d_WilliamsAlligator_1wEMA34_Volume_v1"
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
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Williams Alligator: Smoothed medians (5, 8, 13 periods)
    # Jaw (blue): 13-period SMMA of median price, smoothed 8 bars
    median_price = (high + low) / 2
    sma13 = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(sma13).rolling(window=8, min_periods=8).mean().values  # Smoothed 8
    
    # Teeth (red): 8-period SMMA of median price, smoothed 5 bars
    sma8 = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(sma8).rolling(window=5, min_periods=5).mean().values  # Smoothed 5
    
    # Lips (green): 5-period SMMA of median price, smoothed 3 bars
    sma5 = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(sma5).rolling(window=3, min_periods=3).mean().values  # Smoothed 3
    
    # Calculate 1w EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
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
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Alligator alignment: Teeth > Lips = bullish, Teeth < Lips = bearish
        bullish_alignment = teeth[i] > lips[i]
        bearish_alignment = teeth[i] < lips[i]
        
        # Trend filter: price vs 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > Jaw AND bullish alignment AND uptrend AND volume spike
            if curr_close > jaw[i] and bullish_alignment and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < Jaw AND bearish alignment AND downtrend AND volume spike
            elif curr_close < jaw[i] and bearish_alignment and downtrend and volume_confirm:
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
            # Exit: price breaks below Jaw OR alignment turns bearish OR trend turns down
            elif curr_close < jaw[i] or not bullish_alignment or not uptrend:
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
            # Exit: price breaks above Jaw OR alignment turns bullish OR trend turns up
            elif curr_close > jaw[i] or not bearish_alignment or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals