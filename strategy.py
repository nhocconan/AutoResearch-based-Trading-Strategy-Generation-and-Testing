#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Camarilla pivot levels calculated from previous 4h bar (H4, L4, C4)
# - Long when price breaks above H4 (resistance) with 4h uptrend and volume confirmation
# - Short when price breaks below L4 (support) with 4h downtrend and volume confirmation
# - Uses discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise

name = "1h_4h_camarilla_breakout_v1"
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
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for HTF filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Pre-compute 4h EMA trend filter (50/200)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Pre-compute 4h trend bias (1 for uptrend, -1 for downtrend, 0 for neutral)
    trend_bias = np.zeros(len(ema_50_aligned))
    trend_bias[ema_50_aligned > ema_200_aligned] = 1
    trend_bias[ema_50_aligned < ema_200_aligned] = -1
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_4h = df_4h['volume'].values
    volume_series_4h = pd.Series(volume_4h)
    volume_sma_20_4h = volume_series_4h.rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    # Pre-compute 4h Camarilla pivot levels (based on previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    camarilla_h4 = close_4h + (high_4h - low_4h) * 1.1 / 2
    camarilla_l4 = close_4h - (high_4h - low_4h) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (use previous completed 4h bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(trend_bias[i]) or np.isnan(volume_sma_20_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > camarilla_h4_aligned[i]  # Close above H4 resistance
        breakout_short = price_close < camarilla_l4_aligned[i]  # Close below L4 support
        
        # Trend filter from 4h
        trend_up = trend_bias[i] == 1
        trend_down = trend_bias[i] == -1
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla breakout + uptrend + volume confirmation
        if breakout_long and trend_up and vol_confirm:
            enter_long = True
        
        # Short: Camarilla breakdown + downtrend + volume confirmation
        if breakout_short and trend_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L4 OR trend turns down
            exit_long = (price_close < camarilla_l4_aligned[i]) or (not trend_up)
        elif position == -1:
            # Exit short if price breaks above H4 OR trend turns up
            exit_short = (price_close > camarilla_h4_aligned[i]) or (not trend_down)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals