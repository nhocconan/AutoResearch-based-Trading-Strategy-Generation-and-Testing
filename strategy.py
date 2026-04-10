#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout with 4h/1d regime filter and volume confirmation
# - Primary: 1h timeframe for entry timing precision
# - HTF: 4h for trend direction (EMA50), 1d for volatility regime (ATR percentile)
# - Long: Price breaks above H3 Camarilla pivot (resistance) + 4h EMA50 uptrend + 1d ATR > 30th percentile
# - Short: Price breaks below L3 Camarilla pivot (support) + 4h EMA50 downtrend + 1d ATR > 30th percentile
# - Exit: Price reverts to Camarilla Pivot Point (mean reversion) or opposite H4/L4 break
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Session filter: 08-20 UTC to avoid low-liquidity Asian session
# - Target: 80-120 total trades over 4 years (20-30/year) - within 1h sweet spot
# - Camarilla pivots work well in ranging markets (common in 2025 BTC/ETH bear/range)
# - 4h EMA50 ensures we trade with intermediate-term trend
# - 1d ATR percentile filter avoids low-volatility whipsaws
# - Volume confirmation on breakout increases reliability

name = "1h_4h_1d_camarilla_pivot_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 60 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1h OHLCV
    open_1h = prices['open'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # Pre-compute 4h data
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1h Camarilla Pivot Points (based on previous day)
    # For intraday, we use daily high/low/close to calculate pivots
    # H4 = Close + 1.5*(High-Low)
    # H3 = Close + 1.25*(High-Low)
    # H2 = Close + 1.166*(High-Low)
    # H1 = Close + 1.0833*(High-Low)
    # Pivot = (High+Low+Close)/3
    # L1 = Close - 1.0833*(High-Low)
    # L2 = Close - 1.166*(High-Low)
    # L3 = Close - 1.25*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    
    # We need to align daily OHLC to 1h bars
    # Get previous day's OHLC for each 1h bar
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels for each 1h bar (using previous day's OHLC)
    rng = high_1d_aligned - low_1d_aligned
    h4 = close_1d_aligned + 1.5 * rng
    h3 = close_1d_aligned + 1.25 * rng
    h2 = close_1d_aligned + 1.166 * rng
    h1 = close_1d_aligned + 1.0833 * rng
    pivot = (high_1d_aligned + low_1d_aligned + close_1d_aligned) / 3.0
    l1 = close_1d_aligned - 1.0833 * rng
    l2 = close_1d_aligned - 1.166 * rng
    l3 = close_1d_aligned - 1.25 * rng
    l4 = close_1d_aligned - 1.5 * rng
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 50-day lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1h volume moving average (20-period) for volume confirmation
    volume_ma_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (avoid low-liquidity Asian session)
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 4h trend: price above/below EMA50
        uptrend_4h = close_1h[i] > ema_50_4h_aligned[i]
        downtrend_4h = close_1h[i] < ema_50_4h_aligned[i]
        
        # 1d volatility regime: ATR > 30th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 30
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_spike = volume_1h[i] > 1.5 * volume_ma_20_1h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 resistance + 4h uptrend + vol regime + volume spike
            if (close_1h[i] > h3[i] and uptrend_4h and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.20
            # Short entry: Price breaks below L3 support + 4h downtrend + vol regime + volume spike
            elif (close_1h[i] < l3[i] and downtrend_4h and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Pivot Point (mean reversion)
            # 2. Price breaks opposite H4/L4 level (strong reversal)
            # 3. 4h trend changes
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1h[i] < pivot[i] or  # Reverted to pivot
                    close_1h[i] > h4[i] or     # Break above H4 (take profit)
                    not uptrend_4h             # 4h trend turned down
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1h[i] > pivot[i] or  # Reverted to pivot
                    close_1h[i] < l4[i] or     # Break below L4 (take profit)
                    not downtrend_4h           # 4h trend turned up
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals