# 6H_Triple_Momentum_1wTrend
# 6H_Triple_Momentum_1wTrend
# Combines RSI momentum, MACD histogram, and price action with weekly trend filter
# Uses momentum confluence for high-probability entries in both bull and bear markets
# Weekly trend filter reduces whipsaws and aligns with higher timeframe direction
# Target: 50-150 total trades over 4 years (12-37/year)

#!/usr/bin/env python3
name = "6H_Triple_Momentum_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # RSI(14) momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # MACD(12,26,9) histogram
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Price above/below 20-period EMA for trend bias
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation - volume above 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 26, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(macd_hist[i]) or np.isnan(ema_20[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Momentum conditions
        rsi_bullish = rsi[i] > 50 and rsi[i] > rsi[i-1]
        rsi_bearish = rsi[i] < 50 and rsi[i] < rsi[i-1]
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]
        price_above_ema20 = close[i] > ema_20[i]
        price_below_ema20 = close[i] < ema_20[i]
        volume_confirm = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: bullish momentum confluence in weekly uptrend
            if (rsi_bullish and macd_bullish and price_above_ema20 and 
                volume_confirm and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum confluence in weekly downtrend
            elif (rsi_bearish and macd_bearish and price_below_ema20 and 
                  volume_confirm and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: momentum divergence or weekly trend change
            if (rsi[i] < 50 or macd_hist[i] < 0 or 
                ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: momentum divergence or weekly trend change
            if (rsi[i] > 50 or macd_hist[i] > 0 or 
                ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Triple momentum confluence with weekly trend filter
# - Combines RSI momentum (direction + slope), MACD histogram (momentum + signal crossover)
# - Price position relative to 20-period EMA for trend bias
# - Weekly EMA20 trend filter ensures alignment with higher timeframe direction
# - Volume confirmation (1.5x average) reduces false signals
# - Works in both bull (long signals in weekly uptrend) and bear (short signals in weekly downtrend)
# - Momentum confluence provides higher probability signals than single indicators
# - Exit when momentum diverges or weekly trend changes to avoid whipsaws
# - Position size 0.25 balances capture and risk management
# - Targets 50-150 total trades over 4 years to minimize fee drag while capturing trends
# - Novel combination: RSI slope + MACD histogram slope + price/EMA20 + weekly trend filter not recently tried
# - Designed for 6h timeframe to balance signal frequency with transaction costs
# - Weekly trend filter adapts to market regime (bull/bear) for consistent performance