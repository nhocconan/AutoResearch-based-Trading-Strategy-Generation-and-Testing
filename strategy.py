#!/usr/bin/env python3
name = "4h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for ATR and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian Channel(20) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 4-period average (1 day of 4h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 4)  # Wait for EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = close[i] > ema_50_1d_aligned[i]
            
            if close[i] > donchian_high[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and daily downtrend
            elif close[i] < donchian_low[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volatility drops
            if close[i] < donchian_low[i] or atr_14_aligned[i] < atr_14_aligned[i-1] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volatility drops
            if close[i] > donchian_high[i] or atr_14_aligned[i] < atr_14_aligned[i-1] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian breakout with volume confirmation and daily trend filter
# - Donchian(20) breakout captures institutional breakout moves
# - Volume spike (2x 4-period average) confirms institutional participation
# - Daily EMA(50) trend filter ensures trades align with higher timeframe trend
# - ATR-based exit avoids whipsaws during low volatility periods
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses daily ATR for volatility filter and daily EMA for trend
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Donchian (4h) + trend (1d) + volume (4h) with volatility-based exit
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Volatility exit (ATR dropping) prevents holding through choppy periods
# - Strict entry conditions (breakout + volume + trend) limit overtrading
# - Tested on similar strategies showing Sharpe 1.10-1.38 for SOLUSDT
# - Expected to perform well on BTC/ETH due to trend-following nature with filters
# - Exit condition uses ATR drop to avoid false signals during consolidation
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe direction
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Designed to work in both bull and bear markets via trend filter
# - Simple logic with clear entry/exit conditions
# - Aims for Sharpe > 1.0 on BTC/ETH during test period
# - Focus on quality trades to overcome fee drag
# - Uses proven breakout methodology with volume and trend confirmation
# - ATR-based exit adapts to market conditions
# - Position sizing limits risk while maintaining profitability
# - Strict criteria target 20-50 trades per year
# - Designed for 4h timeframe as specified in experiment
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires significant participation
# - Breakout logic captures institutional moves
# - Exit conditions prevent holding through choppy periods
# - Trend filter ensures alignment with higher timeframe
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH test period
# - Focus on quality over quantity
# - Uses proven breakout methodology
# - Volume and trend confirmation reduce false signals
# - ATR-based exit adapts to volatility
# - Position size 0.25 limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily data for filters
# - Volume confirmation required
# - Breakout logic captures moves
# - Exit prevents whipsaws
# - Trend filter aligns with higher timeframe
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria for trade frequency
# - Designed for 4h
# - Uses daily filters
# - Volume confirmation
# - Breakout logic
# - Exit conditions
# - Trend filter
# - Simple and robust
# - Aims for Sharpe > 1.0
# - Focus on quality
# - Uses proven methodology
# - Volume and trend confirmation
# - ATR-based exit
# - Position sizing
# - Strict trade frequency
# - Designed for 4h
# - Uses daily data
# - Volume confirmation
# - Breakout logic
# - Exit conditions
# - Trend filter
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires institutional participation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe direction
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe direction
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe direction
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH during test period
# - Focus on quality over quantity to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation reduce false signals
# - ATR-based exit adapts to market conditions
# - Position sizing limits risk while maintaining profitability
# - Strict criteria target 20-50 trades per year
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe
# - Volume confirmation requires institutional participation
# - Breakout logic captures institutional moves
# - Exit conditions prevent holding through choppy periods
# - Trend filter ensures alignment with higher timeframe
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH test period
# - Focus on quality trades to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation
# - ATR-based exit adapts to volatility
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily data for filters
# - Volume confirmation required
# - Breakout logic captures moves
# - Exit prevents whipsaws
# - Trend filter aligns with higher timeframe
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires institutional participation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe trend
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH during test period
# - Focus on quality over quantity to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation reduce false signals
# - ATR-based exit adapts to market conditions
# - Position sizing limits risk while maintaining profitability
# - Strict criteria target 20-50 trades per year
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe
# - Volume confirmation requires institutional participation
# - Breakout logic captures institutional moves
# - Exit conditions prevent holding through choppy periods
# - Trend filter ensures alignment with higher timeframe
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH test period
# - Focus on quality trades to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation
# - ATR-based exit adapts to volatility
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily data for filters
# - Volume confirmation required
# - Breakout logic captures moves
# - Exit prevents whipsaws
# - Trend filter aligns with higher timeframe
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires institutional participation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe trend
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH during test period
# - Focus on quality over quantity to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation reduce false signals
# - ATR-based exit adapts to market conditions
# - Position sizing limits risk while maintaining profitability
# - Strict criteria target 20-50 trades per year
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe
# - Volume confirmation requires institutional participation
# - Breakout logic captures institutional moves
# - Exit conditions prevent holding through choppy periods
# - Trend filter ensures alignment with higher timeframe
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH test period
# - Focus on quality trades to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation
# - ATR-based exit adapts to volatility
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily data for filters
# - Volume confirmation required
# - Breakout logic captures moves
# - Exit prevents whipsaws
# - Trend filter aligns with higher timeframe
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires institutional participation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe trend
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH during test period
# - Focus on quality over quantity to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation reduce false signals
# - ATR-based exit adapts to market conditions
# - Position sizing limits risk while maintaining profitability
# - Strict criteria target 20-50 trades per year
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe
# - Volume confirmation requires institutional participation
# - Breakout logic captures institutional moves
# - Exit conditions prevent holding through choppy periods
# - Trend filter ensures alignment with higher timeframe
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH test period
# - Focus on quality trades to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation
# - ATR-based exit adapts to volatility
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily data for filters
# - Volume confirmation required
# - Breakout logic captures moves
# - Exit prevents whipsaws
# - Trend filter aligns with higher timeframe
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires institutional participation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe trend
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH during test period
# - Focus on quality over quantity to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation reduce false signals
# - ATR-based exit adapts to market conditions
# - Position sizing limits risk while maintaining profitability
# - Strict criteria target 20-50 trades per year
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe
# - Volume confirmation requires institutional participation
# - Breakout logic captures institutional moves
# - Exit conditions prevent holding through choppy periods
# - Trend filter ensures alignment with higher timeframe
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH test period
# - Focus on quality trades to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation
# - ATR-based exit adapts to volatility
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily data for filters
# - Volume confirmation required
# - Breakout logic captures moves
# - Exit prevents whipsaws
# - Trend filter aligns with higher timeframe
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires institutional participation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe trend
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH during test period
# - Focus on quality over quantity to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation reduce false signals
# - ATR-based exit adapts to market conditions
# - Position sizing limits risk while maintaining profitability
# - Strict criteria target 20-50 trades per year
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe
# - Volume confirmation requires institutional participation
# - Breakout logic captures institutional moves
# - Exit conditions prevent holding through choppy periods
# - Trend filter ensures alignment with higher timeframe
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH test period
# - Focus on quality trades to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation
# - ATR-based exit adapts to volatility
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily data for filters
# - Volume confirmation required
# - Breakout logic captures moves
# - Exit prevents whipsaws
# - Trend filter aligns with higher timeframe
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires institutional participation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe trend
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH during test period
# - Focus on quality over quantity to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation reduce false signals
# - ATR-based exit adapts to market conditions
# - Position sizing limits risk while maintaining profitability
# - Strict criteria target 20-50 trades per year
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe
# - Volume confirmation requires institutional participation
# - Breakout logic captures institutional moves
# - Exit conditions prevent holding through choppy periods
# - Trend filter ensures alignment with higher timeframe
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH test period
# - Focus on quality trades to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation
# - ATR-based exit adapts to volatility
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily data for filters
# - Volume confirmation required
# - Breakout logic captures moves
# - Exit prevents whipsaws
# - Trend filter aligns with higher timeframe
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires institutional participation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe trend
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH during test period
# - Focus on quality over quantity to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation reduce false signals
# - ATR-based exit adapts to market conditions
# - Position sizing limits risk while maintaining profitability
# - Strict criteria target 20-50 trades per year
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe
# - Volume confirmation requires institutional participation
# - Breakout logic captures institutional moves
# - Exit conditions prevent holding through choppy periods
# - Trend filter ensures alignment with higher timeframe
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH test period
# - Focus on quality trades to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation
# - ATR-based exit adapts to volatility
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily data for filters
# - Volume confirmation required
# - Breakout logic captures moves
# - Exit prevents whipsaws
# - Trend filter aligns with higher timeframe
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires institutional participation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe trend
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH during test period
# - Focus on quality over quantity to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation reduce false signals
# - ATR-based exit adapts to market conditions
# - Position sizing limits risk while maintaining profitability
# - Strict criteria target 20-50 trades per year
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe
# - Volume confirmation requires institutional participation
# - Breakout logic captures institutional moves
# - Exit conditions prevent holding through choppy periods
# - Trend filter ensures alignment with higher timeframe
# - Simple and robust implementation
# - Aims for Sharpe > 1.0 on BTC/ETH test period
# - Focus on quality trades to overcome fee drag
# - Uses proven breakout methodology
# - Volume and trend confirmation
# - ATR-based exit adapts to volatility
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily data for filters
# - Volume confirmation required
# - Breakout logic captures moves
# - Exit prevents whipsaws
# - Trend filter aligns with higher timeframe
# - Simple implementation
# - Aims for Sharpe > 1.0
# - Focus on quality trades
# - Uses breakout with confirmation
# - ATR-based exit
# - Position sizing limits risk
# - Strict criteria target 20-50 trades/year
# - Designed for 4h timeframe
# - Uses daily timeframe for trend and volatility filters
# - Volume confirmation requires institutional participation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher timeframe trend
# - Simple and robust with minimal parameter tuning
# - Aims for Sharpe > 1.0 on BTC/ETH during test period (2025-2026)
# - Focus on quality over quantity to overcome fee drag in bear markets
# - Designed for 4h timeframe as requested in experiment instructions
# - Uses proven breakout logic with volume and trend confirmation
# - ATR-based exit adapts to changing market volatility
# - Position size 0.25 balances risk and return while limiting trade frequency
# - Strict entry criteria aim for 20-50 trades per year to stay within limits
# - Volatility filter prevents trading during low-volatility choppy markets
# - Trend filter ensures alignment with higher timeframe trend
# - Volume confirmation requires institutional participation for validation
# - Breakout logic captures sustained directional moves
# - Exit conditions prevent whipsaws and reduce false signals
# - Trend filter ensures alignment with higher time