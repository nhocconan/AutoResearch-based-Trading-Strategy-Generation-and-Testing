#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND volume > 1.5x 1d volume average AND price > 1w EMA50.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND volume confirmation AND price < 1w EMA50.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Alligator calculated from prior completed 1d bar to avoid look-ahead.
# Volume spike filters low-momentum signals. 1w EMA50 ensures trades only in established weekly trends.
# Works in bull (bullish Alligator alignment with uptrend) and bear (bearish Alligator alignment with downtrend).
# Target: 15-35 trades/year on 1d timeframe.

name = "1d_WilliamsAlligator_1wEMA50_Volume_v1"
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
    
    # Load 1d data ONCE before loop for Alligator and volume filters (primary timeframe data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw (Blue): 13-period SMMA smoothed 8 periods ahead
    # Teeth (Red): 8-period SMMA smoothed 5 periods ahead  
    # Lips (Green): 5-period SMMA smoothed 3 periods ahead
    close_1d = df_1d['close'].values
    
    # SMMA (Smoothed Moving Average) implementation
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        sma = pd.Series(source).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(source, np.nan, dtype=float)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(source)):
            if not np.isnan(sma[i]) and not np.isnan(smma_vals[i-1]):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + sma[i]) / period
            else:
                smma_vals[i] = np.nan
        return smma_vals
    
    jaw = smma(close_1d, 13)  # 13-period SMMA
    teeth = smma(close_1d, 8)  # 8-period SMMA
    lips = smma(close_1d, 5)   # 5-period SMMA
    
    # Smoothed versions (shifted forward)
    jaw_smooth = np.roll(jaw, 8)   # Jaw smoothed 8 periods ahead
    teeth_smooth = np.roll(teeth, 5) # Teeth smoothed 5 periods ahead
    lips_smooth = np.roll(lips, 3)   # Lips smoothed 3 periods ahead
    
    # Align to 1d timeframe (no additional delay needed for Alligator as it's based on completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_smooth)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_smooth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_smooth)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 1w data ONCE before loop for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Alligator, volume, and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
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
        
        # Alligator alignment
        bullish_alignment = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Price vs lips (green line)
        price_above_lips = curr_close > lips_aligned[i]
        price_below_lips = curr_close < lips_aligned[i]
        
        # Trend filter: price vs 1w EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment AND price above lips AND volume spike AND uptrend
            if (bullish_alignment and 
                price_above_lips and 
                volume_spike and 
                uptrend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish Alligator alignment AND price below lips AND volume spike AND downtrend
            elif (bearish_alignment and 
                  price_below_lips and 
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
            # Exit: Alligator alignment turns bearish OR price crosses below lips
            elif (not bullish_alignment) or (curr_close < lips_aligned[i]):
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
            # Exit: Alligator alignment turns bullish OR price crosses above lips
            elif (not bearish_alignment) or (curr_close > lips_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals