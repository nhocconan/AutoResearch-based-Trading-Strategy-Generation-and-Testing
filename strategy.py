#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_market_regime_breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily 20-period EMA for trend direction
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily EMA to 6h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate daily ATR for volatility and regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6-period ATR for 6h timeframe (used for stops)
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly data for longer-term trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return signals
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly ATR for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    atr_1w = pd.Series(tr_w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Volatility regime filter: avoid extremely low volatility environments
    # Use ratio of current 6h ATR to weekly ATR
    atr_ratio_6h_to_1w = atr_6h / (atr_1w_aligned + 1e-10)
    low_vol_filter = atr_ratio_6h_to_1w > 0.3  # Avoid when 6h volatility is too low relative to weekly
    
    # Trend alignment filter: price should be aligned with both daily and weekly trends
    daily_trend_aligned = close > ema_20_1d_aligned
    weekly_trend_aligned = close > ema_50_1w_aligned
    
    # Volume confirmation: current volume above 20-period average
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > 1.3 * vol_ma_20
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_6h[i]) or
            np.isnan(low_vol_filter[i]) or np.isnan(daily_trend_aligned[i]) or
            np.isnan(weekly_trend_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        ema_20 = ema_20_1d_aligned[i]
        ema_50 = ema_50_1w_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_6h_val = atr_6h[i]
        low_vol = low_vol_filter[i]
        daily_trend = daily_trend_aligned[i]
        weekly_trend = weekly_trend_aligned[i]
        vol_confirm = volume_confirmation[i]
        
        # Determine market regime based on volatility and trend alignment
        # In low volatility regimes, we avoid trading
        # In high volatility regimes with trend alignment, we trade breakouts
        
        # Entry conditions - only in favorable regimes
        long_signal = False
        short_signal = False
        
        # Long: price above both EMAs with volume confirmation and adequate volatility
        if (price_close > ema_20 and price_close > ema_50 and 
            vol_confirm and low_vol and daily_trend and weekly_trend):
            long_signal = True
        
        # Short: price below both EMAs with volume confirmation and adequate volatility
        if (price_close < ema_20 and price_close < ema_50 and 
            vol_confirm and low_vol and not daily_trend and not weekly_trend):
            short_signal = True
        
        # Exit conditions - reverse when trend alignment breaks
        exit_long = position == 1 and (price_close < ema_20 or price_close < ema_50)
        exit_short = position == -1 and (price_close > ema_20 or price_close > ema_50)
        
        # Stop loss conditions using 6h ATR
        stop_long = position == 1 and price_low < (entry_price - 2.5 * atr_6h_val)
        stop_short = position == -1 and price_high > (entry_price + 2.5 * atr_6h_val)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6s market regime breakout strategy
# Enters long when price is above both daily (20) and weekly (50) EMA with volume confirmation
# and adequate volatility (6h ATR > 30% of weekly ATR), indicating a strong bullish regime.
# Enters short when price is below both EMAs with volume confirmation and adequate volatility,
# indicating a strong bearish regime.
# Uses volatility regime filter to avoid choppy, low-volatility environments where breakouts fail.
# Exits when price crosses below either EMA (for longs) or above either EMA (for shorts).
# Designed for 6h timeframe to capture medium-term trends while avoiding whipsaws.
# Works in both bull and bear markets by adapting to the prevailing regime.
# Target: 50-150 total trades over 4 years (12-37/year) with selective entry conditions.