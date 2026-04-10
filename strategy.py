#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + volume confirmation + 1w trend filter
# - Long when price breaks above H3 AND volume > 1.5x 20-period average AND 1w close > 1w EMA50 (bullish trend)
# - Short when price breaks below L3 AND volume > 1.5x 20-period average AND 1w close < 1w EMA50 (bearish trend)
# - Exit when price retouches pivot point (PP) with volume confirmation
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivot levels identify intraday support/resistance with high probability reactions
# - Volume confirmation ensures breakouts have institutional conviction
# - 1w EMA50 filter ensures we trade with the higher timeframe trend, reducing counter-trend whipsaws

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h typical price for Camarilla calculation
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    tp_high = prices['high'].values
    tp_low = prices['low'].values
    tp_close = typical_price.values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # H4 = PP + 1.5 * (High - Low)
    # H3 = PP + 1.25 * (High - Low)
    # H2 = PP + 1.166 * (High - Low)
    # H1 = PP + 1.083 * (High - Low)
    # PP = (High + Low + Close) / 3
    # L1 = PP - 1.083 * (High - Low)
    # L2 = PP - 1.166 * (High - Low)
    # L3 = PP - 1.25 * (High - Low)
    # L4 = PP - 1.5 * (High - Low)
    
    # Shift by 1 to use previous bar's high/low/close for current bar's levels
    prev_high = np.concatenate([[np.nan], tp_high[:-1]])
    prev_low = np.concatenate([[np.nan], tp_low[:-1]])
    prev_close = np.concatenate([[np.nan], tp_close[:-1]])
    
    pp = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    h3 = pp + 1.25 * range_hl
    l3 = pp - 1.25 * range_hl
    
    # Pre-compute 12h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w trend filter: EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: close > EMA50 = bullish, close < EMA50 = bearish
    trend_bullish = close_1w > ema_50
    trend_bearish = close_1w < ema_50
    
    # Align HTF indicators to 12h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pp[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND 1w bullish trend
            if (tp_close[i] > h3[i] and 
                volume_spike[i] and 
                trend_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND 1w bearish trend
            elif (tp_close[i] < l3[i] and 
                  volume_spike[i] and 
                  trend_bearish_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price retouches pivot point (PP) with volume confirmation
            exit_long = (position == 1 and 
                        tp_close[i] <= pp[i] and 
                        volume_spike[i])
            exit_short = (position == -1 and 
                         tp_close[i] >= pp[i] and 
                         volume_spike[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals