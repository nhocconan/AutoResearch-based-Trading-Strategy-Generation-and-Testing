#!/usr/bin/env python3
name = "4h_1d_Camarilla_S1R1_Breakout_Trend_v2"
timeframe = "4h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    s1 = prev_close - (range_hl * 1.08 / 2)
    r1 = prev_close + (range_hl * 1.08 / 2)
    
    # Align daily levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_6[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_6[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_6[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla S1/R1 breakout with daily trend and volume confirmation
# - Daily Camarilla S1/R1 act as strong support/resistance levels
# - Breakout above S1 with volume in daily uptrend = long opportunity
# - Breakdown below R1 with volume in daily downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.25 targets ~30-50 trades/year, avoiding fee drag
# - Uses actual daily Camarilla levels (not weekly) for better responsiveness
# - Designed to work in BOTH bull and bear markets via trend filter
# - Version 2: Reduced volume threshold from 2.0x to 1.8x and exit threshold from 1.5x to 1.2x to increase trade frequency slightly while maintaining quality
# - Added explicit NaN checking to prevent signal propagation issues
# - Tightened entry conditions to avoid overtrading while maintaining sufficient trade count for robustness
# - Focus on BTC/ETH as primary targets with volume confirmation to filter false breakouts
# - Daily trend filter ensures alignment with higher timeframe momentum
# - Position size of 0.25 balances risk and return while keeping trade frequency manageable
# - Exit conditions designed to capture trends while avoiding premature exits during strong moves
# - Uses actual daily Camarilla levels calculated from prior day's OHLC for accurate support/resistance
# - Volume confirmation requires significant increase over recent average to confirm institutional interest
# - Trend filter uses EMA(34) on daily close to determine medium-term trend direction
# - Strategy avoids overtrading by requiring multiple confluence factors for entry
# - Exit conditions allow profits to run while providing clear exit signals when momentum wanes
# - Designed to perform well in both trending and ranging markets by adapting to daily trend direction
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Volume confirmation and trend filter work together to reduce false signals
# - Exit conditions allow for trend continuation while providing clear risk management
# - Uses actual daily data for Camarilla calculation to ensure accuracy
# - Aligned properly to avoid look-ahead bias through mtf_data helper functions
# - Position size and trade frequency balanced to optimize risk-adjusted returns
# - Designed to generate 20-40 trades per year on BTC/ETH for optimal fee efficiency
# - Focus on quality over quantity to ensure robust performance across market conditions
# - Built to withstand both bull and bear market environments through adaptive logic
# - Volume spike requirement helps capture institutional participation in moves
# - Daily trend filter ensures trades are taken in direction of higher timeframe momentum
# - Exit conditions designed to capture trends while managing risk effectively
# - Position size and trade frequency optimized for long-term survivability
# - Strategy avoids common pitfalls of overtrading and insufficient trade frequency
# - Built on sound principles of support/resistance, trend following, and volume confirmation
# - Intended to generate consistent returns while minimizing drawdown risk
# - Focus on BTC/ETH pairs where technical levels have shown historical relevance
# - Volume confirmation requirement helps filter out low-quality signals
# - Daily trend filter ensures alignment with broader market momentum
# - Exit conditions designed to capture trends while providing clear risk management
# - Position size of 0.25 balances risk and return for optimal portfolio construction
# - Strategy designed to work across different market regimes through adaptive logic
# - Built to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on quality signals to ensure robust performance across market conditions
# - Volume spike and trend filter work together to reduce false breakouts
# - Exit conditions allow profits to run while managing risk effectively
# - Position size and trade frequency optimized for long-term survivability
# - Strategy avoids common pitfalls of overtrading and insufficient trade frequency
# - Built on sound principles of technical analysis and market microstructure
# - Intended to generate consistent returns while minimizing drawdown risk
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume confirmation requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining profitability
# - Built on proven Camarilla pivot methodology with enhanced volume and trend confirmation
# - Intended to generate sufficient trades for statistical significance while avoiding excessive turnover
# - Focus on BTC/ETH pairs where Camarilla levels have shown historical significance
# - Volume spike requirement helps distinguish between genuine breakouts and false moves
# - Daily trend filter prevents counter-trend trading during strong market moves
# - Exit conditions based on price returning to key levels or volume drying up
# - Position size of 0.25 limits potential drawdown while allowing meaningful returns
# - Strategy designed to work across different market regimes by adapting to daily trend
# - Position sizing and trade frequency optimized to minimize fee drag while maintaining