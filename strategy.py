#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
# - Long when price breaks above 6h Camarilla R4 level + 1d volume > 1.5x 20-period volume average + 1w close > 1w EMA20 (bullish weekly trend)
# - Short when price breaks below 6h Camarilla S4 level + 1d volume > 1.5x 20-period volume average + 1w close < 1w EMA20 (bearish weekly trend)
# - Exit when price reverts to 6h Camarilla middle level (R3/S3 midpoint) or ATR stoploss triggered (adverse move > 2.0*ATR)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Works in bull/bear: Camarilla captures institutional breakout/fade levels; volume confirms participation; weekly trend filter avoids counter-trend trades
# - Target: 12-30 trades/year to stay within fee drag limits while capturing strong moves

name = "6h_1d_1w_camarilla_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 6h data ONCE before loop for Camarilla pivots and ATR (MTF rule compliance)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return signals
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 6h Camarilla pivots (based on previous day's OHLC)
    # Note: For intraday, we use previous 6h bar's OHLC to calculate today's levels
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Camarilla levels: based on previous period's range
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    # Middle = (R3 + S3) / 2
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    # First bar: use current values as fallback
    prev_high[0] = high_6h[0]
    prev_low[0] = low_6h[0]
    prev_close[0] = close_6h[0]
    
    camarilla_r4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    camarilla_s4 = prev_close - 1.5 * (prev_high - prev_low)
    camarilla_middle = (camarilla_r3 + camarilla_s3) / 2.0
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s4)
    camarilla_middle_aligned = align_htf_to_ltf(prices, df_6h, camarilla_middle)
    
    # Pre-compute 6h ATR(20) for stoploss
    tr1 = pd.Series(high_6h).rolling(2).max() - pd.Series(low_6h).rolling(2).min()
    tr2 = abs(pd.Series(high_6h).shift(1) - pd.Series(close_6h))
    tr3 = abs(pd.Series(low_6h).shift(1) - pd.Series(close_6h))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_6h, atr_20)
    
    # Pre-compute 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_middle_aligned[i]) or np.isnan(atr_20_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume average (moderate threshold)
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Weekly trend filter: close > EMA20 for bullish, close < EMA20 for bearish
        weekly_close = df_1w['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
        weekly_bullish = weekly_close_aligned[i] > ema_20_1w_aligned[i]
        weekly_bearish = weekly_close_aligned[i] < ema_20_1w_aligned[i]
        
        # Camarilla breakout conditions
        breakout_r4 = close_price > camarilla_r4_aligned[i]
        breakdown_s4 = close_price < camarilla_s4_aligned[i]
        
        # Entry conditions
        enter_long = breakout_r4 and vol_confirm and weekly_bullish
        enter_short = breakdown_s4 and vol_confirm and weekly_bearish
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (close_price < camarilla_middle_aligned[i] or  # Revert to middle level
                     close_price < entry_price - 2.0 * atr_20_aligned[i]))  # ATR stoploss
        exit_short = (position == -1 and 
                     (close_price > camarilla_middle_aligned[i] or  # Revert to middle level
                      close_price > entry_price + 2.0 * atr_20_aligned[i]))  # ATR stoploss
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
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