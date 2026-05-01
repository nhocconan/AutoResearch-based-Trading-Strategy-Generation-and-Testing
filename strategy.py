#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA50 trend + volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1d EMA50 AND volume > 1.3x 4h volume average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1d EMA50 AND volume > 1.3x 4h volume average.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Williams Alligator identifies trend phases; EMA50 filters for higher-timeframe direction; volume confirms momentum.
# Works in bull (catch Alligator wake-up in uptrend) and bear (catch Alligator wake-up in downtrend).
# Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years).

name = "4h_WilliamsAlligator_1dEMA50_Volume_v1"
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
    
    # Calculate Williams Alligator (SMMA with specific periods)
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 1d EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume average (20-period)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Alligator, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.3x 4h volume average
        if vol_ma_4h[i] <= 0 or np.isnan(vol_ma_4h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_4h[i] * 1.3)
        
        # Trend filter: price vs 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Williams Alligator signals
        bullish_alligator = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alligator = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator AND uptrend AND volume confirmation
            if bullish_alligator and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish Alligator AND downtrend AND volume confirmation
            elif bearish_alligator and downtrend and volume_confirm:
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
            # Exit: Alligator turns bearish OR trend turns down
            elif not bullish_alligator or (not uptrend):
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
            # Exit: Alligator turns bullish OR trend turns up
            elif not bearish_alligator or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals