#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator system with 1w EMA50 trend filter and volume confirmation.
# Long when Alligator jaw < teeth < lips (bullish alignment) AND price > lips AND close > 1w EMA50 AND volume > 1.5x 20-period volume median.
# Short when Alligator jaw > teeth > lips (bearish alignment) AND price < lips AND close < 1w EMA50 AND volume > 1.5x 20-period volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Williams Alligator identifies trend phases via smoothed medians; 1w EMA50 filters for primary trend alignment; volume spike confirms conviction.
# Designed for low trade frequency on 12h timeframe to minimize fee drag while capturing sustained moves.

name = "12h_WilliamsAlligator_1wEMA50_Volume_v1"
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
    
    # Calculate Williams Alligator (using SMMA - smoothed moving average)
    # Jaw: SMMA(13, 8) of median price
    # Teeth: SMMA(8, 5) of median price  
    # Lips: SMMA(5, 3) of median price
    median_price = (high + low) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average - similar to EMA but with different smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # SMMA(13, 8) - period=13
    teeth = smma(median_price, 8)  # SMMA(8, 5) - period=8
    lips = smma(median_price, 5)   # SMMA(5, 3) - period=5
    
    # Calculate 1w EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Alligator, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Alligator alignment: bullish (jaw < teeth < lips) or bearish (jaw > teeth > lips)
        bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Price vs lips (Alligator's lips act as dynamic support/resistance)
        price_above_lips = curr_close > lips[i]
        price_below_lips = curr_close < lips[i]
        
        # Trend filter: price vs 1w EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator AND price > lips AND uptrend AND volume spike
            if bullish_alignment and price_above_lips and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: bearish Alligator AND price < lips AND downtrend AND volume spike
            elif bearish_alignment and price_below_lips and downtrend and volume_confirm:
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
            # Exit: Alligator turns bearish OR price < lips OR trend turns down
            elif not bullish_alignment or price_below_lips or not uptrend:
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
            # Exit: Alligator turns bullish OR price > lips OR trend turns up
            elif not bearish_alignment or price_above_lips or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals