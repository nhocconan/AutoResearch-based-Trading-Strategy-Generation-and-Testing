#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX Trend Filter with 4h/1d Confluence for BTC/ETH
# - Primary: 1h timeframe for entry timing with strict ADX(14) > 25 trend filter
# - HTF: 4h for trend direction (EMA20 cross), 1d for volatility regime (ATR > 30th percentile)
# - Long: ADX>25 + DI+ > DI- + price > EMA20(1h) + 4h EMA50 uptrend + 1d ATR > 30th percentile
# - Short: ADX>25 + DI- > DI+ + price < EMA20(1h) + 4h EMA50 downtrend + 1d ATR > 30th percentile
# - Exit: ADX < 20 (trend weakening) or opposite DI crossover
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Session filter: 08-20 UTC to avoid low-liquidity Asian session
# - Target: 60-120 total trades over 4 years (15-30/year) - within 1h sweet spot
# - ADX filters out ranging markets (common in 2025 BTC/ETH bear/range)
# - 4h EMA50 ensures intermediate-term trend alignment
# - 1d ATR percentile filter avoids low-volatility whipsaws
# - Only trade strong trending conditions to minimize false signals and fee drag

name = "1h_4h_1d_adx_trend_v1"
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
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1h EMA(20) for trend filter
    ema_20_1h = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1h ADX(14) for trend strength
    # True Range
    tr1 = pd.Series(high_1h).shift(1) - pd.Series(low_1h).shift(1)
    tr2 = abs(pd.Series(high_1h) - pd.Series(close_1h).shift(1))
    tr3 = abs(pd.Series(low_1h) - pd.Series(close_1h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1h) - pd.Series(high_1h).shift(1)
    down_move = pd.Series(low_1h).shift(1) - pd.Series(low_1h)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di_1h = 100 * plus_dm_smooth / atr_1h
    minus_di_1h = 100 * minus_dm_smooth / atr_1h
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1h - minus_di_1h) / (plus_di_1h + minus_di_1h)
    adx_1h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1_1d = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2_1d = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3_1d = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 50-day lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Session filter: 08-20 UTC (avoid low-liquidity Asian session)
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(adx_1h[i]) or np.isnan(plus_di_1h[i]) or np.isnan(minus_di_1h[i]) or 
            np.isnan(ema_20_1h[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1h trend strength: ADX > 25
        strong_trend = adx_1h[i] > 25
        
        # 1h trend direction: DI+ > DI- for long, DI- > DI+ for short
        bullish_di = plus_di_1h[i] > minus_di_1h[i]
        bearish_di = minus_di_1h[i] > plus_di_1h[i]
        
        # 1h price vs EMA20: price above/below EMA for confirmation
        price_above_ema = close_1h[i] > ema_20_1h[i]
        price_below_ema = close_1h[i] < ema_20_1h[i]
        
        # 4h trend: price above/below EMA50
        uptrend_4h = close_1h[i] > ema_50_4h_aligned[i]
        downtrend_4h = close_1h[i] < ema_50_4h_aligned[i]
        
        # 1d volatility regime: ATR > 30th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 30
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Strong trend + bullish DI + price above EMA20 + 4h uptrend + vol regime
            if (strong_trend and bullish_di and price_above_ema and uptrend_4h and vol_regime):
                position = 1
                signals[i] = 0.20
            # Short entry: Strong trend + bearish DI + price below EMA20 + 4h downtrend + vol regime
            elif (strong_trend and bearish_di and price_below_ema and downtrend_4h and vol_regime):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. ADX < 20 (trend weakening)
            # 2. Opposite DI crossover (trend reversal)
            # 3. Price crosses EMA20 in opposite direction
            
            if position == 1:  # Long position
                exit_condition = (
                    adx_1h[i] < 20 or              # Trend weakening
                    minus_di_1h[i] > plus_di_1h[i] or  # DI crossover bearish
                    close_1h[i] < ema_20_1h[i]     # Price below EMA20
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                exit_condition = (
                    adx_1h[i] < 20 or              # Trend weakening
                    plus_di_1h[i] > minus_di_1h[i] or  # DI crossover bullish
                    close_1h[i] > ema_20_1h[i]     # Price above EMA20
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals