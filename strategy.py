# 4H_FUNDING_RATE_MEAN_REVERSION_ZSCORE_30D_12HTREND
# Hypothesis: Funding rate mean-reversion provides edge in BTC/ETH. Extreme positive funding (longs paying shorts) 
# precedes short-term reversals down; extreme negative funding precedes reversals up. 
# Uses 30-day z-score of funding rate to detect extremes. Filters by 12h EMA50 trend to avoid fighting trend.
# Works in bull (fades overextended longs) and bear (fades oversold shorts). Low turnover (~20-40 trades/year).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load funding rate data for the symbol (assuming BTC/ETH/USDT)
    symbol = getattr(prices, 'symbol', 'BTCUSDT').replace('USDT', '')
    funding_path = f"/var/lib/temp/data/processed/funding/{symbol}.parquet"
    try:
        funding_df = pd.read_parquet(funding_path)
        # Align funding timestamps to price timestamps
        funding_df = funding_df.set_index('funding_time')
        # Reindex to match price index
        funding_aligned = funding_df.reindex(prices['open_time'], method='ffill')
        funding_rate = funding_aligned['funding_rate'].values
    except:
        # Fallback: use zero funding if file not found (should not happen in backtest)
        funding_rate = np.zeros(n)
    
    # Calculate 30-day z-score of funding rate
    # 30 days = 90 funding intervals (8h each)
    window = 90
    funding_mean = np.full(n, np.nan)
    funding_std = np.full(n, np.nan)
    
    for i in range(window, n):
        window_data = funding_rate[i-window:i]
        funding_mean[i] = np.mean(window_data)
        funding_std[i] = np.std(window_data)
    
    # Z-score: (current - mean) / std
    funding_zscore = np.full(n, np.nan)
    valid_mask = (~np.isnan(funding_mean)) & (~np.isnan(funding_std)) & (funding_std > 0)
    funding_zscore[valid_mask] = (funding_rate[valid_mask] - funding_mean[valid_mask]) / funding_std[valid_mask]
    
    # Get 12h data for trend filter: EMA(50) on close
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h_50 = np.full(len(df_12h), np.nan)
    for i in range(len(close_12h)):
        if i < 49:
            ema_12h_50[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            alpha = 2 / (50 + 1)
            ema_12h_50[i] = close_12h[i] * alpha + ema_12h_50[i-1] * (1 - alpha)
    
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup period
    start_idx = max(window, 50)
    
    for i in range(start_idx, n):
        if np.isnan(funding_zscore[i]) or np.isnan(ema_12h_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        zscore = funding_zscore[i]
        trend = ema_12h_50_aligned[i]
        prev_trend = ema_12h_50_aligned[i-1] if i > 0 else trend
        
        if position == 0:
            # Long when funding extremely negative (shorts paying longs) AND trend not strongly down
            if zscore < -2.0 and trend >= prev_trend:
                signals[i] = 0.25
                position = 1
            # Short when funding extremely positive (longs paying shorts) AND trend not strongly up
            elif zscore > 2.0 and trend <= prev_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when funding normalizes or trend turns down
            if zscore > -0.5 or trend < prev_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when funding normalizes or trend turns up
            if zscore < 0.5 or trend > prev_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4H_FUNDING_RATE_MEAN_REVERSION_ZSCORE_30D_12HTREND"
timeframe = "4h"
leverage = 1.0