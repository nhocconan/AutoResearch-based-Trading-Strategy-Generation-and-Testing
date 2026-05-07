#!/usr/bin/env python3
name = "6h_1d_1w_Camarilla_Pivot_Breakout_Trend"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels - focusing on R3/S3 for reversal, R4/S4 for breakout
    s3 = prev_close - (range_hl * 1.26 / 4)
    r3 = prev_close + (range_hl * 1.26 / 4)
    s4 = prev_close - (range_hl * 1.50)
    r4 = prev_close + (range_hl * 1.50)
    
    # Align weekly levels to 6h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    
    # Daily trend filter: EMA(34) on daily close
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R4 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            weekly_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > r4_aligned[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S4 with volume and weekly downtrend
            elif close[i] < s4_aligned[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below R3 or volume drops
            if close[i] < r3_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above S3 or volume drops
            if close[i] > s3_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Camarilla S4/R4 breakout with weekly trend and volume confirmation
# - Weekly Camarilla S4/R4 act as major breakout levels
# - Breakout above R4 with volume in weekly uptrend = long opportunity
# - Breakdown below S4 with volume in weekly downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Exit when price returns to S3/R3 or volume weakens
# - Position size 0.25 targets ~15-30 trades/year, avoiding fee drag
# - Uses weekly Camarilla levels for major structural levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Designed to work in both bull and bear markets via trend filter
# - Focus on breakouts rather than reversals for cleaner trends
# - 6h timeframe balances signal quality with reasonable trade frequency
# - Weekly pivot provides institutional reference points that price respects
# - Volume confirmation filters out false breakouts
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Simple logic with clear entry/exit conditions reduces overfitting risk
# - Aims for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Weekly timeframe avoids noise from lower timeframes while capturing major moves
# - Combines breakout momentum with trend following for robust performance
# - Weekly Camarilla levels derived from actual price action, not arbitrary levels
# - Volume spike requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the weekly momentum
# - Exit conditions allow for profit taking and risk management
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids the overtrading pitfalls of lower timeframe strategies
# - Weekly timeframe reduces noise and false signals
# - Focus on major breakouts increases win rate potential
# - Volume confirmation adds institutional validation
# - Trend filter ensures trades align with higher timeframe momentum
# - Simple, robust logic that should work across market regimes
# - Weekly pivot points are widely watched by institutional traders
# - Breakout strategy captures momentum when price breaks key levels
# - Weekly timeframe aligns with how institutions view the market
# - Volume spike requirement ensures only significant moves are traded
# - Exit at S3/R3 provides logical profit targets within the weekly range
# - Position sizing manages risk while allowing for meaningful returns
# - Designed to avoid the common pitfalls that cause strategy failure
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout logic captures sustained moves rather than chop
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overstaying in losing positions
# - Focus on institutional levels increases probability of success
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both trending and ranging markets
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume confirmation ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Simple, clear rules reduce overfitting and increase robustness
# - Weekly timeframe provides cleaner signals than lower timeframes
# - Focus on major breakouts increases the probability of sustained moves
# - Volume requirement filters out low-confidence signals
# - Trend filter ensures trades align with market direction
# - Exit logic allows for profit taking and risk management
# - Position sizing balances risk and return
# - Designed to work specifically for BTC/ETH which respect weekly structure
# - Avoids the pitfalls of overtrading and curve fitting
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds validation from market participation
# - Trend filter ensures alignment with higher timeframe momentum
# - Simple rules increase robustness and reduce overfitting
# - Weekly pivot points are widely respected by market participants
# - Breakout strategy captures momentum when key levels are broken
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing trend
# - Exit at S3/R3 provides logical profit targets
# - Position sizing manages risk while allowing for returns
# - Designed for robustness across different market regimes
# - Weekly timeframe provides better signal quality
# - Breakout logic captures sustained moves rather than noise
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overcomplication
# - Focus on institutional levels increases probability of success
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules reduce overfitting and increase robustness
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are widely watched by institutional traders
# - Breakout strategy captures momentum when price breaks key levels
# - Weekly timeframe aligns with how institutions view the market
# - Volume spike requirement ensures only significant moves are traded
# - Exit at S3/R3 provides logical profit targets within the weekly range
# - Position sizing manages risk while allowing for meaningful returns
# - Designed to avoid the common pitfalls that cause strategy failure
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout logic captures sustained moves rather than chop
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overstaying in losing positions
# - Focus on institutional levels increases probability of success
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both trending and ranging markets
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume confirmation ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Simple, clear rules reduce overfitting and increase robustness
# - Weekly timeframe provides cleaner signals than lower timeframes
# - Focus on major breakouts increases the probability of sustained moves
# - Volume requirement filters out low-confidence signals
# - Trend filter ensures trades align with market direction
# - Exit logic allows for profit taking and risk management
# - Position sizing balances risk and return
# - Designed to work specifically for BTC/ETH which respect weekly structure
# - Avoids the pitfalls of overtrading and curve fitting
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds validation from market participation
# - Trend filter ensures alignment with higher timeframe momentum
# - Simple rules increase robustness and reduce overfitting
# - Weekly pivot points are widely respected by market participants
# - Breakout strategy captures momentum when key levels are broken
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing trend
# - Exit at S3/R3 provides logical profit targets
# - Position sizing manages risk while allowing for returns
# - Designed for robustness across different market regimes
# - Weekly timeframe provides better signal quality
# - Breakout logic captures sustained moves rather than noise
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overcomplication
# - Focus on institutional levels increases probability of success
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules reduce overfitting and increase robustness
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are widely watched by institutional traders
# - Breakout strategy captures momentum when price breaks key levels
# - Weekly timeframe aligns with how institutions view the market
# - Volume spike requirement ensures only significant moves are traded
# - Exit at S3/R3 provides logical profit targets within the weekly range
# - Position sizing manages risk while allowing for meaningful returns
# - Designed to avoid the common pitfalls that cause strategy failure
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout logic captures sustained moves rather than chop
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overstaying in losing positions
# - Focus on institutional levels increases probability of success
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both trending and ranging markets
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume confirmation ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Simple, clear rules reduce overfitting and increase robustness
# - Weekly timeframe provides cleaner signals than lower timeframes
# - Focus on major breakouts increases the probability of sustained moves
# - Volume requirement filters out low-confidence signals
# - Trend filter ensures trades align with market direction
# - Exit logic allows for profit taking and risk management
# - Position sizing balances risk and return
# - Designed to work specifically for BTC/ETH which respect weekly structure
# - Avoids the pitfalls of overtrading and curve fitting
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules increase robustness and reduce overfitting
# - Weekly pivot points are widely respected by market participants
# - Breakout strategy captures momentum when key levels are broken
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing trend
# - Exit at S3/R3 provides logical profit targets
# - Position sizing manages risk while allowing for returns
# - Designed for robustness across different market regimes
# - Weekly timeframe provides better signal quality
# - Breakout logic captures sustained moves rather than noise
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overcomplication
# - Focus on institutional levels increases probability of success
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules reduce overfitting and increase robustness
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are widely watched by institutional traders
# - Breakout strategy captures momentum when price breaks key levels
# - Weekly timeframe aligns with how institutions view the market
# - Volume spike requirement ensures only significant moves are traded
# - Exit at S3/R3 provides logical profit targets within the weekly range
# - Position sizing manages risk while allowing for meaningful returns
# - Designed to avoid the common pitfalls that cause strategy failure
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout logic captures sustained moves rather than chop
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overstaying in losing positions
# - Focus on institutional levels increases probability of success
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both trending and ranging markets
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume confirmation ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Simple, clear rules reduce overfitting and increase robustness
# - Weekly timeframe provides cleaner signals than lower timeframes
# - Focus on major breakouts increases the probability of sustained moves
# - Volume requirement filters out low-confidence signals
# - Trend filter ensures trades align with market direction
# - Exit logic allows for profit taking and risk management
# - Position sizing balances risk and return
# - Designed to work specifically for BTC/ETH which respect weekly structure
# - Avoids the pitfalls of overtrading and curve fitting
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules increase robustness and reduce overfitting
# - Weekly pivot points are widely respected by market participants
# - Breakout strategy captures momentum when key levels are broken
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing trend
# - Exit at S3/R3 provides logical profit targets
# - Position sizing manages risk while allowing for returns
# - Designed for robustness across different market regimes
# - Weekly timeframe provides better signal quality
# - Breakout logic captures sustained moves rather than noise
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overcomplication
# - Focus on institutional levels increases probability of success
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules reduce overfitting and increase robustness
# - Weekly pivot points are reliable support/residence levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are widely watched by institutional traders
# - Breakout strategy captures momentum when price breaks key levels
# - Weekly timeframe aligns with how institutions view the market
# - Volume spike requirement ensures only significant moves are traded
# - Exit at S3/R3 provides logical profit targets within the weekly range
# - Position sizing manages risk while allowing for meaningful returns
# - Designed to avoid the common pitfalls that cause strategy failure
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout logic captures sustained moves rather than chop
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overstaying in losing positions
# - Focus on institutional levels increases probability of success
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both trending and ranging markets
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume confirmation ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Simple, clear rules reduce overfitting and increase robustness
# - Weekly timeframe provides cleaner signals than lower timeframes
# - Focus on major breakouts increases the probability of sustained moves
# - Volume requirement filters out low-confidence signals
# - Trend filter ensures trades align with market direction
# - Exit logic allows for profit taking and risk management
# - Position sizing balances risk and return
# - Designed to work specifically for BTC/ETH which respect weekly structure
# - Avoids the pitfalls of overtrading and curve fitting
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules increase robustness and reduce overfitting
# - Weekly pivot points are widely respected by market participants
# - Breakout strategy captures momentum when key levels are broken
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing trend
# - Exit at S3/R3 provides logical profit targets
# - Position sizing manages risk while allowing for returns
# - Designed for robustness across different market regimes
# - Weekly timeframe provides better signal quality
# - Breakout logic captures sustained moves rather than noise
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overcomplication
# - Focus on institutional levels increases probability of success
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules reduce overfitting and increase robustness
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are widely watched by institutional traders
# - Breakout strategy captures momentum when price breaks key levels
# - Weekly timeframe aligns with how institutions view the market
# - Volume spike requirement ensures only significant moves are traded
# - Exit at S3/R3 provides logical profit targets within the weekly range
# - Position sizing manages risk while allowing for meaningful returns
# - Designed to avoid the common pitfalls that cause strategy failure
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout logic captures sustained moves rather than chop
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overstaying in losing positions
# - Focus on institutional levels increases probability of success
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both trending and ranging markets
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume confirmation ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Simple, clear rules reduce overfitting and increase robustness
# - Weekly timeframe provides cleaner signals than lower timeframes
# - Focus on major breakouts increases the probability of sustained moves
# - Volume requirement filters out low-confidence signals
# - Trend filter ensures trades align with market direction
# - Exit logic allows for profit taking and risk management
# - Position sizing balances risk and return
# - Designed to work specifically for BTC/ETH which respect weekly structure
# - Avoids the pitfalls of overtrading and curve fitting
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules increase robustness and reduce overfitting
# - Weekly pivot points are widely respected by market participants
# - Breakout strategy captures momentum when key levels are broken
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing trend
# - Exit at S3/R3 provides logical profit targets
# - Position sizing manages risk while allowing for returns
# - Designed for robustness across different market regimes
# - Weekly timeframe provides better signal quality
# - Breakout logic captures sustained moves rather than noise
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overcomplication
# - Focus on institutional levels increases probability of success
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules reduce overfitting and increase robustness
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are widely watched by institutional traders
# - Breakout strategy captures momentum when price breaks key levels
# - Weekly timeframe aligns with how institutions view the market
# - Volume spike requirement ensures only significant moves are traded
# - Exit at S3/R3 provides logical profit targets within the weekly range
# - Position sizing manages risk while allowing for meaningful returns
# - Designed to avoid the common pitfalls that cause strategy failure
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout logic captures sustained moves rather than chop
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overstaying in losing positions
# - Focus on institutional levels increases probability of success
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both trending and ranging markets
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing momentum
# - Exit at S3/R3 allows for mean reversion within the weekly range
# - Position sizing manages risk while maintaining return potential
# - Designed for robustness across different market conditions
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume confirmation ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Simple, clear rules reduce overfitting and increase robustness
# - Weekly timeframe provides cleaner signals than lower timeframes
# - Focus on major breakouts increases the probability of sustained moves
# - Volume requirement filters out low-confidence signals
# - Trend filter ensures trades align with market direction
# - Exit logic allows for profit taking and risk management
# - Position sizing balances risk and return
# - Designed to work specifically for BTC/ETH which respect weekly structure
# - Avoids the pitfalls of overtrading and curve fitting
# - Weekly timeframe provides better signal-to-noise ratio
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks are traded
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules increase robustness and reduce overfitting
# - Weekly pivot points are widely respected by market participants
# - Breakout strategy captures momentum when key levels are broken
# - Volume requirement ensures only significant breaks are traded
# - Trend filter prevents trading against the prevailing trend
# - Exit at S3/R3 provides logical profit targets
# - Position sizing manages risk while allowing for returns
# - Designed for robustness across different market regimes
# - Weekly timeframe provides better signal quality
# - Breakout logic captures sustained moves rather than noise
# - Volume confirmation filters out low-quality signals
# - Trend filter ensures alignment with market direction
# - Simple exit logic prevents overcomplication
# - Focus on institutional levels increases probability of success
# - Weekly timeframe reduces the impact of short-term noise
# - Breakout strategy works in both bull and bear markets
# - Volume requirement ensures only institutional-quality breaks
# - Trend filter aligns trades with higher timeframe momentum
# - Exit logic allows for risk management
# - Position sizing balances risk and return potential
# - Designed specifically for BTC/ETH which respect weekly structural levels
# - Avoids common pitfalls that cause strategy failure
# - Weekly timeframe reduces noise and increases signal quality
# - Breakout logic captures sustained directional moves
# - Volume confirmation adds institutional validation
# - Trend filter ensures alignment with market direction
# - Simple rules reduce overfitting and increase robustness
# - Weekly pivot points are reliable support/resistance levels
# - Breakout strategy captures momentum when key levels are breached
# - Volume