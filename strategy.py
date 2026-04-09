#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for trend direction and 1d RSI for mean reversion timing
# - Uses 4h HTF for Donchian channel (20-period): breakout above/below determines trend
# - Uses 1d HTF for RSI(14): extreme readings (<30 or >70) signal mean reversion entries
# - In bullish 4h trend (price > upper Donchian): look for long entries when 1d RSI < 30 (oversold pullback)
# - In bearish 4h trend (price < lower Donchian): look for short entries when 1d RSI > 70 (overbought bounce)
# - Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity periods
# - Fixed position size 0.20 to control drawdown and minimize fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)

name = "1h_4h_1d_donchian_rsi_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channel (20 periods)
    period20_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    
    # Calculate 1d RSI (14 periods)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align all HTF data to 1h timeframe (wait for completed HTF bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Trend determination from 4h Donchian
        bullish_4h_trend = close[i] > donchian_upper_aligned[i]
        bearish_4h_trend = close[i] < donchian_lower_aligned[i]
        
        # RSI extremes from 1d
        oversold_1d = rsi_aligned[i] < 30
        overbought_1d = rsi_aligned[i] > 70
        
        # Fixed position size
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit conditions
            if bullish_4h_trend:
                # In bullish 4h trend: exit when overbought or trend changes to bearish
                if overbought_1d or bearish_4h_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Not in bullish 4h trend: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if bearish_4h_trend:
                # In bearish 4h trend: exit when oversold or trend changes to bullish
                if oversold_1d or bullish_4h_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Not in bearish 4h trend: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on trend and RSI extremes
            if bullish_4h_trend and oversold_1d:
                # In bullish 4h trend, 1d oversold: long mean reversion
                position = 1
                signals[i] = position_size
            elif bearish_4h_trend and overbought_1d:
                # In bearish 4h trend, 1d overbought: short mean reversion
                position = -1
                signals[i] = -position_size
    
    return signals