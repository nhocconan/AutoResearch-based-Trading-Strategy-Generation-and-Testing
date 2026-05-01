#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) and price > lips, with 1w EMA50 uptrend and volume > 1.5x 20-bar average.
# Short when Alligator jaws > teeth > lips (bearish alignment) and price < lips, with 1w EMA50 downtrend and volume confirmation.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Alligator smoothed with SMMA (5,8,13 periods) as per Bill Williams.
# Works in bull (trend continuation) and bear (trend continuation) regimes by following 1w EMA50 trend.
# Target trades: 20-60 total over 4 years (5-15/year) to minimize fee drag.

name = "1d_WilliamsAlligator_1wTrend_Volume_v2"
timeframe = "1d"
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
    
    # Calculate Williams Alligator (SMMA: Smoothed Moving Average)
    def smma(data, period):
        # Smoothed Moving Average: first value is SMA, then recursive smoothing
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(data, np.nan, dtype=float)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(data)):
            if not np.isnan(sma[i]) and not np.isnan(smma_vals[i-1]):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + data[i]) / period
        return smma_vals
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Alligator (need at least 13 periods)
    start_idx = 13
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(ema_50_1w_aligned[i])):
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
        
        # Alligator alignment conditions
        bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Price relative to lips
        price_above_lips = curr_close > lips[i]
        price_below_lips = curr_close < lips[i]
        
        # 1w trend filter: EMA50 slope (using current vs 5 periods ago)
        if i >= 5:
            ema_now = ema_50_1w_aligned[i]
            ema_past = ema_50_1w_aligned[i-5]
            if not np.isnan(ema_now) and not np.isnan(ema_past):
                trend_up = ema_now > ema_past
                trend_down = ema_now < ema_past
            else:
                trend_up = False
                trend_down = False
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish alignment AND price above lips AND 1w uptrend AND volume confirmation
            if (bullish_alignment and 
                price_above_lips and 
                trend_up and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bearish alignment AND price below lips AND 1w downtrend AND volume confirmation
            elif (bearish_alignment and 
                  price_below_lips and 
                  trend_down and 
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
            # Exit: Alligator alignment turns bearish OR price crosses below lips
            elif (not bullish_alignment) or (curr_close < lips[i]):
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
            # Exit: Alligator alignment turns bullish OR price crosses above lips
            elif (not bearish_alignment) or (curr_close > lips[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals