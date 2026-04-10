#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band squeeze breakout with 4h trend filter and volume confirmation
# - Primary: 1h Bollinger Band width below 20th percentile (squeeze) + breakout above/below bands
# - HTF trend: 4h close > 50-period EMA for long bias, < 50-period EMA for short bias
# - HTF volume: 1d volume > 1.3x 20-period MA for participation confirmation
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Long: BB squeeze + price > upper BB + 4h EMA uptrend + volume spike + session
# - Short: BB squeeze + price < lower BB + 4h EMA downtrend + volume spike + session
# - Exit: price returns to 20-period SMA (mean reversion) or opposite band touch
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# - Works in bull/bear: BB squeeze captures low volatility breakouts, 4h EMA filter avoids counter-trend, volume confirms

name = "1h_4h_1d_bb_squeeze_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_4h = df_4h['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1h Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20  # Normalized BB width
    
    # Calculate BB width percentile (20-period lookback for squeeze detection)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    is_squeeze = bb_width_percentile < 20  # BB width in lowest 20%
    
    # Calculate 4h EMA(50)
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(sma_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(is_squeeze[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: BB squeeze + price > upper BB + 4h EMA uptrend + volume spike + session
            if (is_squeeze[i] and close[i] > upper_band[i] and close[i] > ema_4h_aligned[i] and volume_confirm):
                position = 1
                signals[i] = 0.20
            # Short entry: BB squeeze + price < lower BB + 4h EMA downtrend + volume spike + session
            elif (is_squeeze[i] and close[i] < lower_band[i] and close[i] < ema_4h_aligned[i] and volume_confirm):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price returns to 20-period SMA or touches opposite band
            if position == 1:  # Long position
                if close[i] <= sma_20[i] or close[i] >= upper_band[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if close[i] >= sma_20[i] or close[i] <= lower_band[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals