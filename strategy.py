#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
# Long when Alligator lines are bullish (jaw < teeth < lips) with price > lips, volume > 1.5x 1d volume average, and price > 1d EMA34.
# Short when Alligator lines are bearish (jaw > teeth > lips) with price < lips, volume confirmation, and price < 1d EMA34.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Alligator calculated from 6h data with proper min_periods to avoid look-ahead.
# 1d EMA and volume filters loaded once before loop using mtf_data helpers.
# Works in bull (Alligator alignment with uptrend) and bear (Alligator alignment with downtrend) regimes.
# Target: 12-37 trades/year on 6h timeframe.

name = "6h_WilliamsAlligator_1dEMA34_Volume_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop for EMA and volume filters (HTF)
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
    
    # Calculate Williams Alligator on 6h data
    # Jaw: Blue line (13-period SMMA, shifted 8 bars ahead)
    # Teeth: Red line (8-period SMMA, shifted 5 bars ahead)
    # Lips: Green line (5-period SMMA, shifted 3 bars ahead)
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    # We'll use EMA as approximation for SMMA as it's commonly done
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Shift the lines as per Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Set NaN for shifted values that don't have enough data
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Alligator, and EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 1.5x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_1d_aligned[i] * 1.5)
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Alligator conditions
        # Bullish: jaw < teeth < lips (lines separated and ascending)
        bullish_alligator = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        # Bearish: jaw > teeth > lips (lines separated and descending)
        bearish_alligator = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator AND price > lips AND volume spike AND uptrend
            if (bullish_alligator and 
                curr_close > lips[i] and 
                volume_spike and 
                uptrend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish Alligator AND price < lips AND volume spike AND downtrend
            elif (bearish_alligator and 
                  curr_close < lips[i] and 
                  volume_spike and 
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
            # Exit: Alligator turns bearish OR price < lips OR trend reverses
            elif (not bullish_alligator) or (curr_close < lips[i]) or (curr_close < ema_34_1d_aligned[i]):
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
            # Exit: Alligator turns bullish OR price > lips OR trend reverses
            elif (not bearish_alligator) or (curr_close > lips[i]) or (curr_close > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals