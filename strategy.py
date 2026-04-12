#!/usr/bin/env python3
"""
1d_1w_MeanReversion_With_FundingRate_ZScore
Hypothesis: Use weekly funding rate Z-score as a contrarian signal for daily mean reversion.
Long when funding rate Z-score < -2 (extremely negative funding = bullish sentiment extreme),
short when Z-score > +2 (extremely positive funding = bearish sentiment extreme).
Filter trades with daily RSI extremes (RSI < 30 for long, RSI > 70 for short) and 
volume > 1.5x average to confirm conviction. Exit when funding Z-score reverts toward zero
or RSI reaches neutral zone (40-60). Designed for low frequency (<15 trades/year) 
with high conviction in BTC/ETH mean reversion during funding extremes.
Works in bull markets (catching oversold bounces) and bear markets (selling overbought rallies).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_MeanReversion_With_FundingRate_ZScore"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY FUNDING RATE DATA (simulated via proxy: weekly price change as funding proxy) ===
    # Note: In real implementation, this would load from data/processed/funding/*.parquet
    # For this simulation, we use weekly log returns as a proxy for funding rate sentiment
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly log returns as funding proxy
    funding_proxy = np.diff(np.log(close_1w), prepend=np.log(close_1w[0]))
    # Calculate Z-score of funding proxy (30-week lookback)
    funding_ma = pd.Series(funding_proxy).rolling(window=30, min_periods=30).mean().values
    funding_std = pd.Series(funding_proxy).rolling(window=30, min_periods=30).std().values
    funding_zscore = np.where(funding_std > 0, (funding_proxy - funding_ma) / funding_std, 0)
    funding_zscore_aligned = align_htf_to_ltf(prices, df_1w, funding_zscore)
    
    # === DAILY INDICATORS ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    loss_ma = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(loss_ma > 0, gain_ma / loss_ma, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(funding_zscore_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # Funding Z-score extremes
        funding_extreme_neg = funding_zscore_aligned[i] < -2.0
        funding_extreme_pos = funding_zscore_aligned[i] > 2.0
        funding_reverting = np.abs(funding_zscore_aligned[i]) < 0.5
        
        # Entry logic
        long_signal = (funding_extreme_neg and rsi_oversold and strong_volume)
        short_signal = (funding_extreme_pos and rsi_overbought and strong_volume)
        
        # Exit logic
        exit_long = (position == 1 and 
                    (funding_reverting or rsi_neutral[i]))
        exit_short = (position == -1 and 
                     (funding_reverting or rsi_neutral[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals