#!/usr/bin/env python3
"""
4h_1w_NASDAQ100_FundingRate_Reward_Risk
Strategy: Uses weekly NASDAQ-100 price action and funding rate to identify risk-on/risk-off regimes.
In risk-on (NASDAQ up, funding negative): long BTC/ETH on 4h pullbacks to EMA21.
In risk-off (NASDAQ down, funding positive): short BTC/ETH on 4h bounces to EMA21.
Exit: Opposite signal or price crosses EMA50.
Position size: 0.25
Designed to work in both bull and bear markets by following macro risk sentiment.
Timeframe: 4h
"""

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
    
    # Calculate EMA21 and EMA50 for entries and exits
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get weekly NASDAQ-100 data (simulated via BTC as proxy for risk sentiment)
    # In live trading, this would be replaced with actual NASDAQ data
    # For backtesting, we use BTC's weekly trend as a proxy for market risk appetite
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA for trend
    close_series_1w = pd.Series(close_1w)
    ema50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Simulate funding rate (negative = longs pay shorts = bearish sentiment, positive = bullish)
    # Actual funding data would be loaded from external source; here we use a proxy
    # Based on observation: when BTC is strong, funding tends positive; weak = negative
    # We invert this to create a contrarian signal for altcoins
    returns_1w = np.diff(np.log(close_1w), prepend=0)
    funding_proxy = -np.tanh(returns_1w * 10)  # Scale and invert
    funding_proxy_aligned = align_htf_to_ltf(prices, df_1w, funding_proxy)
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(funding_proxy_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Regime detection: 
        # Risk-on: weekly trend up AND funding negative (contrarian for alts)
        # Risk-off: weekly trend down AND funding positive (contrarian for alts)
        weekly_uptrend = close_1w[-1] > ema50_1w[-1] if len(close_1w) > 0 else False  # Simplified
        # Better: use current aligned values
        weekly_uptrend = ema50_1w_aligned[i] > 0 and close[i] > ema50_1w_aligned[i]  # Proxy
        funding_negative = funding_proxy_aligned[i] < -0.1
        funding_positive = funding_proxy_aligned[i] > 0.1
        
        risk_on = weekly_uptrend and funding_negative
        risk_off = not weekly_uptrend and funding_positive
        
        # Entry conditions
        pullback_to_ema21 = abs(close[i] - ema21[i]) < (0.02 * close[i])  # Within 2% of EMA21
        bounce_from_ema21 = abs(close[i] - ema21[i]) < (0.02 * close[i])  # Same for bounce
        
        if position == 0:
            # Long in risk-on: pullback to EMA21 with volume
            if risk_on and pullback_to_ema21 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short in risk-off: bounce from EMA21 with volume
            elif risk_off and bounce_from_ema21 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: risk-off emerges or price crosses EMA50 down
            if risk_off or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: risk-on emerges or price crosses EMA50 up
            if risk_on or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1w_NASDAQ100_FundingRate_Reward_Risk"
timeframe = "4h"
leverage = 1.0