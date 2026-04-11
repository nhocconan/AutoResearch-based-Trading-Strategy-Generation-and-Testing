#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion + 1w trend filter + volume confirmation
# - Williams %R(14) from 1d: long when < -80 (oversold), short when > -20 (overbought)
# - 1w EMA(34) trend filter: only long when price > EMA34, only short when price < EMA34
# - Volume confirmation: current volume > 1.5x 20-period 1d volume SMA (avoid low-volume fakes)
# - ATR stoploss: exit when price moves 2.5 * ATR(14) against position
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits for 1d
# - Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)
# - Williams %R is a proven mean-reversion oscillator that works well on daily timeframe

name = "1d_1w_williamsr_volume_trend_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-compute 1d Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Pre-compute 1d volume SMA(20) for confirmation
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d ATR(14) for dynamic stoploss
    tr1 = pd.Series(high).shift(1) - pd.Series(low).shift(1)
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).shift(1) - pd.Series(close).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Track entry price for stoploss
    entry_price = 0.0
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_14_1d[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Williams %R conditions
        wr_oversold = williams_r[i] < -80
        wr_overbought = williams_r[i] > -20
        
        # 1w trend filter: price > EMA34 for uptrend, price < EMA34 for downtrend
        uptrend = price_close > ema_34_1w_aligned[i]
        downtrend = price_close < ema_34_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Williams %R oversold + uptrend + volume confirmation
        if wr_oversold and uptrend and vol_confirm:
            enter_long = True
        
        # Short: Williams %R overbought + downtrend + volume confirmation
        if wr_overbought and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Williams %R extreme or ATR stoploss
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R becomes overbought OR ATR stoploss hit
            exit_long = williams_r[i] > -20
            # ATR stoploss: exit if price drops 2.5 * ATR below entry price
            if entry_price > 0 and price_close < entry_price - 2.5 * atr_14_1d[i]:
                exit_long = True
        elif position == -1:
            # Exit short if Williams %R becomes oversold OR ATR stoploss hit
            exit_short = williams_r[i] < -80
            # ATR stoploss: exit if price rises 2.5 * ATR above entry price
            if entry_price > 0 and price_close > entry_price + 2.5 * atr_14_1d[i]:
                exit_short = True
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals