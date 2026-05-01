#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
# Uses 1h timeframe for entry timing, 4h for trend direction, and daily HTF for regime filtering.
# Long when price breaks above Camarilla R3 AND price > 4h EMA34 AND daily close > weekly EMA8 (bull regime).
# Short when price breaks below Camarilla S3 AND price < 4h EMA34 AND daily close < weekly EMA8 (bear regime).
# Uses discrete sizing 0.20 to minimize fee churn. ATR-based stoploss: signal→0 when price moves against position by 1.5*ATR.
# Target: 15-30 trades/year (60-120 total over 4 years) to stay within fee drag limits.
# Camarilla levels provide precise intraday support/resistance, EMA34 offers smooth trend filter, regime filter avoids counter-trend trades.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Regime_v1"
timeframe = "1h"
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
    
    # Calculate 1h Camarilla levels from previous 1h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # We use previous bar's OHLC to avoid look-ahead
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    camarilla_r3 = prev_close + 1.125 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.125 * (prev_high - prev_low)
    
    # Calculate 4h EMA34 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate daily regime filter: daily close vs weekly EMA8
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 8:
        return np.zeros(n)
    
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 8:
        return np.zeros(n)
    
    ema_8_1w = pd.Series(df_1w['close'].values).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    # Daily close > weekly EMA8 = bull regime, < = bear regime
    daily_close = df_1d['close'].values
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    bull_regime = daily_close_aligned > ema_8_1w_aligned
    bear_regime = daily_close_aligned < ema_8_1w_aligned
    
    # Calculate 1h volume median (20-period for stability)
    vol_median_1h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Start after warmup for ATR, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(daily_close_aligned[i]) or 
            np.isnan(ema_8_1w_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_median_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.3x 1h volume median
        if vol_median_1h[i] <= 0 or np.isnan(vol_median_1h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_1h[i] * 1.3)
        
        # Trend filter: price vs 4h EMA34
        uptrend = curr_close > ema_34_4h_aligned[i]
        downtrend = curr_close < ema_34_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Camarilla R3 AND uptrend AND bull regime AND volume confirmation
            if (curr_high > camarilla_r3[i] and 
                uptrend and 
                bull_regime[i] and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: Break below Camarilla S3 AND downtrend AND bear regime AND volume confirmation
            elif (curr_low < camarilla_s3[i] and 
                  downtrend and 
                  bear_regime[i] and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 1.5*ATR
            if curr_close < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Camarilla S3 OR trend turns down OR regime turns bearish
            elif (curr_low < camarilla_s3[i]) or (not uptrend) or (not bull_regime[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 1.5*ATR
            if curr_close > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Camarilla R3 OR trend turns up OR regime turns bullish
            elif (curr_high > camarilla_r3[i]) or (not downtrend) or (bull_regime[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
    
    return signals