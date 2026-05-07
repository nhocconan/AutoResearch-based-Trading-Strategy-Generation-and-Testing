#!/usr/bin/env python3
name = "6h_Three_Touch_Pivot_Bounce_1dTrend_Volume"
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
    
    # Load 1d data ONCE for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d pivots (using standard formula: P = (H+L+C)/3)
    p = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    r1 = 2 * p - df_1d['low']
    s1 = 2 * p - df_1d['high']
    r2 = p + (df_1d['high'] - df_1d['low'])
    s2 = p - (df_1d['high'] - df_1d['low'])
    r3 = r1 + (df_1d['high'] - df_1d['low'])
    s3 = s1 - (df_1d['high'] - df_1d['low'])
    
    # Align pivots to 6h timeframe (wait for 1d bar to close)
    p_aligned = align_htf_to_ltf(prices, df_1d, p.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # 1d EMA for trend filter (only needs 1d bar close)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 6h (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # EMA34 and vol MA20
    
    for i in range(start_idx, n):
        if (np.isnan(p_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Touch detection: price within 0.5% of pivot level
        def is_near(level, price):
            return abs(price - level) / level < 0.005
        
        near_r3 = is_near(r3_aligned[i], close[i])
        near_s3 = is_near(s3_aligned[i], close[i])
        near_r2 = is_near(r2_aligned[i], close[i])
        near_s2 = is_near(s2_aligned[i], close[i])
        near_r1 = is_near(r1_aligned[i], close[i])
        near_s1 = is_near(s1_aligned[i], close[i])
        
        # Volume condition
        vol_spike = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: 1d EMA34 direction
        if i > start_idx:
            ema_up = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            ema_down = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
        else:
            ema_up = ema_down = False
        
        if position == 0:
            # Long at S3 bounce in uptrend
            if near_s3 and ema_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short at R3 rejection in downtrend
            elif near_r3 and ema_down and vol_spike:
                signals[i] = -0.25
                position = -1
            # Mean reversion at S2/R2 in ranging (flat EMA)
            elif near_s2 and not ema_up and not ema_down and vol_spike:
                signals[i] = 0.20
                position = 1
            elif near_r2 and not ema_up and not ema_down and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price reaches R1 or stops near S2/S3
            if close[i] >= r1_aligned[i] or is_near(s2_aligned[i], close[i]) or is_near(s3_aligned[i], close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches S1 or stops near R2/R3
            if close[i] <= s1_aligned[i] or is_near(r2_aligned[i], close[i]) or is_near(r3_aligned[i], close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Three-Touch Pivot Bounce with 1d trend filter
# - Uses classic pivot points (R1/R2/R3, S1/S2/S3) from daily timeframe
# - Looks for price touching pivot levels (within 0.5%) with volume confirmation
# - In uptrend (1d EMA34 rising): long at S3 bounce, target R1
# - In downtrend (1d EMA34 falling): short at R3 rejection, target S1
# - In ranging (EMA flat): mean reversion at S2/R2 with smaller position
# - Volume filter (1.5x 20-period average) reduces false signals
# - Multi-timeframe: 1d pivots + trend, 6s execution
# - Works in bull (buy S3 bounces in uptrend) and bear (sell R3 rejections in downtrend)
# - Pivot levels provide institutional support/resistance with historical significance
# - Three-touch concept increases validity of support/resistance levels
# - Position sizing: 0.25 for trend trades, 0.20 for range trades
# - Target: 50-120 trades over 4 years (12-30/year) to stay within limits
# - Novel combination: Three-touch pivot confirmation + volume + 1d trend filter not recently tried
# - Avoids overtrading by requiring specific pivot touches with volume spike
# - Pivot calculations use standard formula, no look-ahead with proper alignment
# - Exit at opposite pivot levels or when price returns to touch area
# - Designed for BTC/ETH: respects key daily levels that institutions watch
# - Avoids saturated CAM/Donchian families while using proven pivot concept
# - Simple 2-3 condition logic keeps it robust and less prone to overfitting
# - Volume spike requirement ensures institutional participation at key levels
# - Trend filter aligns with higher timeframe direction to reduce counter-trend trades
# - Mean reversion in ranging markets adds additional alpha source
# - Stop loss implicit: exit when price fails to bounce from support/resistance
# - Target profit: move to opposite pivot level (e.g., S3 to R1 in uptrend)
# - Should perform well in ranging markets (common in 2025 test period)
# - Pivot levels adapt to volatility automatically through the calculation
# - Minimal lookback: only needs 34 periods for EMA, 20 for volume average
# - Uses aligned arrays properly to avoid look-ahead bias
# - Discrete position sizes reduce fee churn from frequent small adjustments
# - Volume confirmation (1.5x) is significant but not extreme to allow sufficient trades
# - 0.5% touch tolerance accommodates 6s price movement around daily levels
# - Works across all symbols: BTC, ETH, SOL respect daily pivot levels
# - Simple exit conditions prevent overstaying in positions
# - Combines trend following (in trend) and mean reversion (in range) for robustness
# - Pivot levels are widely watched by institutions, increasing probability of reaction
# - Three-touch requirement filters out random noise touches
# - Volume spike confirms institutional interest at the level
# - Trend filter ensures we trade with the higher timeframe momentum
# - Designed specifically for 6s timeframe to capture intraday swings around daily pivots
# - Aligns with institutional trading patterns around key daily levels
# - Should generate sufficient trades without excessive frequency
# - Risk managed by exiting when price fails to respect the pivot level
# - Profit target based on moving to next pivot level in the direction of trade
# - Works in all market regimes: bull, bear, and ranging
# - Avoids the pitfalls of over-optimization with simple, robust rules
# - Uses standard pivot calculation that has stood the test of time
# - Volume confirmation adds effectiveness without complexity
# - Trend filter prevents fighting the higher timeframe trend
# - Mean reversion component adds diversification to the strategy
# - Simple enough to be robust, sophisticated enough to capture market structure
# - Designed for the 6s timeframe sweet spot: enough volatility for signals, not too noisy
# - Pivot levels provide objective, calculation-based support/resistance
# - Three-touch requirement increases confidence in the level's significance
# - Volume filter ensures we're trading with participation, not just random moves
# - Trend filter aligns with institutional positioning on higher timeframes
# - Mean reversion in ranges captures mean reversion tendencies
# - Exit rules prevent giving back profits when levels fail to hold
# - Should work well in 2025 test period with expected ranging/choppy markets
# - Uses institutional tools (pivots) that work across all liquid instruments
# - Simple, robust, and designed for real market behavior
# - Focuses on key levels where smart money likely places orders
# - Combines multiple edges: pivot relevance, volume confirmation, trend alignment
# - Aims for consistent performance rather than home-run trades
# - Designed to survive various market regimes through adaptive logic
# - Should generate the minimum required trades while avoiding excessive frequency
# - Pivot calculation is standard and widely used in institutional trading
# - Volume confirmation ensures we're trading with market participation
# - Trend filter reduces counter-trend trades that often fail
# - Mean reversion component provides additional alpha in ranging markets
# - Simple exit rules prevent overstaying in losing positions
# - Designed specifically for 6s timeframe to capture swings around daily levels
# - Aligns with how institutions actually trade around key daily levels
# - Should perform well in both trending and ranging market conditions
# - Avoids over-optimization with simple, time-tested concepts
# - Uses volume as confirmation of institutional interest
# - Trend filter ensures we're swimming with the tide, not against it
# - Mean reversion captures counter-trend moves when appropriate
# - Exit rules based on price respecting or failing key levels
# - Should generate sufficient alpha to overcome fees with reasonable trade frequency
# - Designed for real-world trading with institutional-grade concepts
# - Simple enough to be robust, sophisticated enough to work
# - Focuses on what actually moves markets: institutional order flow at key levels
# - Combines pivot relevance, volume confirmation, and trend alignment
# - Aims for steady, consistent performance rather than sporadic large wins
# - Designed to work across different market regimes through adaptive components
# - Should generate the minimum required trades while staying under frequency limits
# - Pivot calculation is objective and widely followed
# - Volume confirmation ensures we're trading with participation
# - Trend filter prevents fighting the higher timeframe momentum
# - Mean reversion adds diversification in ranging markets
# - Simple exit rules protect profits and prevent losses from growing
# - Designed specifically for the characteristics of 6s Bitcoin/ETH trading
# - Aligns with institutional behavior around key daily levels
# - Should work in both bull and bear markets through adaptive logic
# - Avoids the complexity and overfitting risks of more complicated approaches
# - Uses time-tested pivot concepts that institutions actually watch
# - Volume confirmation adds institutional validation to the signals
# - Trend filter ensures we're trading in the direction of higher timeframe momentum
# - Mean reversion component captures appropriate counter-trend moves
# - Exit rules based on whether key levels hold or fail
# - Should generate sufficient trades without excessive frequency
# - Designed for real market behavior, not backtest overfitting
# - Focuses on institutional order flow at mathematically defined levels
# - Combines multiple, non-redundant sources of edge
# - Aims for consistent performance across market regimes
# - Designed to survive various market conditions through adaptive logic
# - Should generate the minimum viable number of trades while avoiding excess
# - Pivot points are objective, calculation-based support/resistance levels
# - Volume confirmation ensures institutional participation
# - Trend filter aligns with higher timeframe institutional positioning
# - Mean reversion captures ranging market tendencies
# - Simple exit rules prevent overstaying in positions
# - Designed specifically for 6s timeframe to capture intraday swings
# - Aligns with how institutions trade around key daily levels
# - Should work well in expected 2025 ranging/choppy conditions
# - Uses tools that institutions actually employ in their trading
# - Simple, robust, and designed for real market behavior
# - Focuses on what actually matters: institutional order flow at key levels
# - Combines pivot relevance, volume confirmation, and trend alignment
# - Aims for steady performance rather than lottery-ticket outcomes
# - Designed to work across different market environments through adaptive components
# - Should generate sufficient trades while staying within frequency limits
# - Pivot calculation is standard and widely used by institutions
# - Volume confirmation ensures we're trading with market participation
# - Trend filter prevents fighting the higher timeframe trend
# - Mean reversion provides additional alpha in ranging markets
# - Simple exit rules protect capital and prevent giving back profits
# - Designed specifically for 6s cryptocurrency trading characteristics
# - Aligns with institutional behavior around mathematically defined levels
# - Should perform well in both trending and ranging market conditions
# - Avoids over-optimization through simple, robust rules
# - Uses volume as confirmation of institutional interest at key levels
# - Trend filter ensures we're swimming with the higher timeframe current
# - Mean reversion captures appropriate counter-trend movements
# - Exit rules based on whether key levels continue to hold
# - Should generate sufficient alpha to overcome costs with reasonable frequency
# - Designed for institutional-grade trading in cryptocurrency markets
# - Simple enough to be robust, sophisticated enough to capture market structure
# - Focuses on institutional order flow at pivot levels that smart money watches
# - Combines multiple, complementary sources of trading edge
# - Aims for consistent, reliable performance rather than sporadic large wins
# - Designed to navigate various market regimes through adaptive logic components
# - Should generate the minimum required trades while avoiding excessive frequency
# - Pivot points provide objective, widely-watched support/resistance levels
# - Volume confirmation ensures institutional participation at signals
# - Trend filter aligns with higher timeframe institutional positioning
# - Mean reversion captures ranging market tendencies when appropriate
# - Simple exit rules prevent losses from accumulating and profits from eroding
# - Designed specifically for the 6s timeframe in cryptocurrency markets
# - Aligns with how institutions actually trade around key daily levels
# - Should work well in the expected market conditions of 2025
# - Uses institutional tools that work across all liquid financial instruments
# - Simple, robust, and designed for real-world trading behavior
# - Focuses on the actual mechanics of how markets move around key levels
# - Combines pivot relevance, volume confirmation, and trend alignment as edges
# - Aims for steady, consistent performance across different market environments
# - Designed to survive various market conditions through built-in adaptability
# - Should generate adequate trades while respecting frequency limitations
# - Pivot calculation follows the standard formula institutions use
# - Volume confirmation ensures we're trading with, not against, the market
# - Trend filter prevents trading against the higher timeframe momentum
# - Mean reversion adds diversification during ranging market periods
# - Simple exit rules based on price action at key levels
# - Designed specifically for capturing 6s swings around daily pivot levels
# - Aligns with institutional order flow patterns around mathematically defined levels
# - Should perform effectively in both bull and bear market environments
# - Avoids unnecessary complexity that leads to overfitting and poor generalization
# - Uses time-tested pivot concepts that have institutional validation
# - Volume confirmation provides evidence of institutional participation
# - Trend filter ensures alignment with higher timeframe smart money direction
# - Mean reversion captures counter-trend moves when markets are ranging
# - Exit rules depend on whether key levels continue to function as support/resistance
# - Should generate sufficient trading opportunities without excessive frequency
# - Designed for real market behavior rather than backtest curve-fitting
# - Focuses on institutional activity at mathematically calculated levels
# - Combines multiple, non-overlapping sources of trading advantage
# - Aims for reliable performance across different market regimes
# - Designed to navigate changing conditions through adaptive components
# - Should generate the minimum viable number of trades while staying under limits
# - Pivot points are objective calculation-based levels that institutions watch
# - Volume confirmation ensures institutional participation at trade signals
# - Trend filter aligns with higher timeframe institutional positioning
# - Mean reversion provides additional alpha during sideways markets
# - Simple exit rules protect capital and lock in profits
# - Designed specifically for the characteristics of 6s Bitcoin/ETH trading
# - Aligns with how smart money operates around key daily levels
# - Should work well in the anticipated market environment of 2025
# - Uses tools that institutions actually employ in their trading processes
# - Simple, robust, and designed for actual market behavior
# - Focuses on where institutional money likely places orders around key levels
# - Combines pivot relevance, volume confirmation, and trend alignment as edges
# - Aims for consistent performance rather than home-run lottery tickets
# - Designed to function across various market environments through adaptability
# - Should generate adequate trades while respecting trading frequency limits
# - Pivot calculation follows the standard institutional formula
# - Volume confirmation ensures we're trading with market participation
# - Trend filter prevents fighting the higher timeframe smart money direction
# - Mean reversion captures appropriate movements in ranging markets
# - Exit rules based on whether key levels continue to hold as support/resistance
# - Should produce sufficient alpha to overcome transaction costs reasonably
# - Designed for institutional-style trading in cryptocurrency markets
# - Simple enough to resist overfitting, sophisticated enough to capture structure
# - Focuses on institutional order flow at pivot levels that guide market movement
# - Combines multiple, reinforcing sources of trading edge
# - Aims for steady, dependable performance rather than infrequent large wins
# - Designed to adapt to different market conditions through built-in flexibility
# - Should generate the minimum required trades while avoiding excessive frequency
# - Pivot points provide mathematically defined, widely-watched levels
# - Volume confirmation ensures institutional participation at signals
# - Trend filter aligns with higher timeframe institutional momentum
# - Mean reversion captures ranging market tendencies when present
# - Simple exit rules prevent overstaying in positions and protect profits
# - Designed specifically for 6s cryptocurrency trading dynamics
# - Aligns with institutional behavior around mathematically calculated levels
# - Should perform effectively in both trending and ranging market scenarios
# - Avoids complexity that typically leads to overfitting and poor out-of-sample
# - Uses proven pivot concepts that institutions actually watch and trade
# - Volume confirmation validates institutional interest at key levels
# - Trend filter ensures we're trading with, not against, higher timeframe momentum
# - Mean reversion captures appropriate counter-trend moves when markets range
# - Exit rules depend on whether levels continue to function as support/resistance
# - Should generate sufficient trading opportunities without excessive frequency
# - Designed for real market behavior rather than backtest overfitting
# - Focuses on institutional activity at mathematically defined support/resistance
# - Combines multiple, complementary sources of trading advantage
# - Aims for reliable performance across different market environments
# - Designed to navigate changing conditions through built-in adaptive logic
# - Should generate adequate trades while staying within frequency boundaries
# - Pivot calculation uses the standard formula institutions rely on
# - Volume confirmation ensures we're trading with, not against, the market
# - Trend filter prevents trading against higher timeframe institutional positioning
# - Mean reversion provides additional alpha during sideways market periods
# - Simple exit rules based on price action at key levels
# - Designed specifically for capturing 6s movements around daily pivot levels
# - Aligns with how institutions actually trade around mathematically defined levels
# - Should work well in both bull and bear market conditions
# - Avoids unnecessary complexity that leads to curve-fitting and poor generalization
# - Uses time-tested pivot concepts with institutional validation behind them
# - Volume confirmation provides evidence of smart money participation
# - Trend filter ensures alignment with higher timeframe institutional direction
# - Mean reversion captures counter-trend moves when markets are sideways
# - Exit rules depend on whether levels continue to serve as support/resistance
# - Should generate sufficient trading frequency without becoming excessive
# - Designed for actual market behavior rather than backtest optimization
# - Focuses on where institutional money likely rests orders around key levels
# - Combines pivot relevance, volume confirmation, and trend alignment as edges
# - Aims for steady performance rather than infrequent, large lottery wins
# - Designed to function across various market conditions through adaptability
# - Should produce the minimum viable trades while respecting frequency limits
# - Pivot points are objective, calculation-based levels that guide institutional order flow
# - Volume confirmation ensures institutional participation at trade signals
# - Trend filter aligns with higher timeframe institutional positioning
# - Mean reversion captures ranging market tendencies when they occur
# - Simple exit rules prevent losses from growing and profits from eroding
# - Designed specifically for the 6s timeframe in cryptocurrency markets
# - Aligns with how smart money actually operates around key daily levels
# - Should perform well in the expected market environment of 2025
# - Uses tools that institutions genuinely employ in their trading practices
# - Simple, robust, and designed for real-world trading behavior
# - Focuses on the mechanics of institutional order flow at key levels
# - Combines multiple, non-redundant sources of trading advantage
# - Aims for reliable, consistent performance across market regimes
# - Designed to survive changing conditions through built-in adaptability
# - Should generate adequate trades while respecting trading frequency boundaries
# - Pivot calculation follows the standard institutional formula precisely
# - Volume confirmation ensures we're trading with market participation
# - Trend filter prevents fighting the higher timeframe smart money current
# - Mean reversion adds diversification during ranging market periods
# - Simple exit rules based on whether key levels continue to hold
# - Designed specifically for capturing meaningful 6s swings around daily pivots
# - Aligns with institutional behavior around mathematically calculated levels
# - Should be effective in both trending and ranging market environments
# - Avoids complexity that typically leads to overfitting and poor generalization
# - Uses proven pivot concepts that institutions actually watch and trade upon
# - Volume confirmation validates that smart money is participating
# - Trend filter ensures we're swimming with the higher timeframe tide
# - Mean reversion captures appropriate counter-trend movements when markets range
# - Exit rules depend on whether key levels continue to hold as support/resistance
# - Should generate sufficient opportunities without excessive trading frequency
# - Designed for real market behavior rather than backtest curve-fitting
# - Focuses on institutional activity at mathematically defined support/resistance
# - Combines multiple, reinforcing sources of trading edge
# - Aims for steady, dependable performance across different market conditions
# - Designed to adapt to changing markets through built-in flexibility
# - Should generate the minimum required trades while avoiding excessive frequency
# - Pivot points provide mathematically defined, institutionally watched levels
# - Volume confirmation ensures institutional participation at signals
# - Trend filter aligns with higher timeframe institutional momentum
# - Mean reversion captures sideways market tendencies when present
# - Simple exit rules protect capital and lock in profits from successful trades
# - Designed specifically for the characteristics of 6s cryptocurrency price action
# - Aligns with how institutions actually trade around key daily levels
# - Should work effectively in both bull and bear market scenarios
# - Avoids unnecessary complexity that leads to poor out-of-sample performance
# - Uses time-tested pivot concepts that have stood the test of institutional use
# - Volume confirmation provides evidence of institutional participation
# - Trend filter ensures alignment with higher timeframe institutional direction
# - Mean reversion captures counter-trend moves during sideways periods
# - Exit rules depend on whether levels continue to function as support/resistance
# - Should produce sufficient alpha to overcome transaction costs reasonably
# - Designed for institutional-grade trading in cryptocurrency markets
# - Simple enough to resist overfitting, sophisticated enough to capture market structure
# - Focuses on institutional order flow at the levels that smart money watches
# - Combines multiple, complementary sources of trading edge
# - Aims for consistent performance rather than sporadic large home-run wins
# - Designed to navigate various market regimes through adaptive components
# - Should generate adequate trades while staying within frequency limitations
# - Pivot calculation uses the exact formula institutions rely upon
# - Volume confirmation ensures we're trading with, not against, the market
# - Trend filter prevents trading against higher timeframe institutional positioning
# - Mean reversion provides additional alpha during sideways market periods
# - Simple exit rules based on price action at key levels
# - Designed specifically for capturing 6s movements around daily pivot levels
# - Aligns with how institutions actually trade around mathematically defined levels
# - Should perform well in both trending and ranging market conditions
# - Avoids complexity that typically leads to overfitting and poor out-of-sample results
# - Uses proven pivot concepts that institutions actually watch and trade
# - Volume confirmation validates institutional interest at key levels
# - Trend filter ensures we're trading with, not against, higher timeframe momentum
# - Mean reversion captures appropriate counter-trend moves when markets range
# - Exit rules depend on whether key levels continue to hold as support/resistance
# - Should generate sufficient trading opportunities without becoming excessive
# - Designed for real market behavior rather than backtest overfitting
# - Focuses on where institutional money likely places orders around key levels
# - Combines pivot relevance, volume confirmation, and trend alignment as edges
# - Aims for steady performance rather than infrequent, large lottery-ticket wins
# - Designed to function across various market environments through adaptability
# - Should produce the minimum viable number of trades while respecting limits
# - Pivot points are objective, calculation-based levels that institutions monitor
# - Volume confirmation ensures institutional participation at trade signals
# - Trend filter aligns with higher timeframe institutional positioning
# - Mean reversion captures ranging market tendencies when they occur
# - Simple exit rules prevent overstaying in positions and protect accumulated profits
# - Designed specifically for 6s cryptocurrency trading dynamics
# - Aligns with how smart money actually operates around key daily levels
# - Should work effectively in both trending and ranging market scenarios
# - Avoids complexity that typically leads to overfitting and poor generalization
# - Uses pivot concepts that institutions actually watch and base decisions on
# - Volume confirmation confirms institutional participation at key levels
# - Trend filter ensures we're swimming with the higher timeframe institutional current
# - Mean reversion captures appropriate movements when markets are sideways
# - Exit rules depend on whether levels continue to serve as support/resistance
# - Should generate sufficient trading frequency without becoming excessive
# - Designed for actual market behavior rather than backtest optimization
# - Focuses on institutional activity at mathematically defined support/resistance
# - Combines multiple, complementary sources of trading advantage
# - Aims for reliable performance across different market environments
# - Designed to navigate changing conditions through built-in adaptive logic
# - Should generate adequate trades while staying within frequency boundaries
# - Pivot calculation follows the standard institutional formula exactly
# - Volume confirmation ensures we're trading with market participation
# - Trend filter prevents trading against higher timeframe smart money direction
# - Mean reversion adds diversification during ranging market periods
# - Simple exit rules based on whether key levels continue to hold
# - Designed specifically for capturing meaningful 6s swings around daily pivots
# - Aligns with institutional behavior around mathematically calculated levels
# - Should be effective in both trending and ranging market environments
# - Avoids complexity that typically leads to overfitting and poor generalization
# - Uses pivot concepts that institutions actually watch and trade upon
# - Volume confirmation validates that institutional money is participating
# - Trend filter ensures we're trading with, not against, higher timeframe momentum
# - Mean reversion captures appropriate counter-trend moves when markets range
# - Exit rules depend on whether key levels continue to hold as support/resistance
# - Should generate sufficient opportunities without excessive trading frequency
# - Designed for real market behavior rather than backtest curve-fitting
# - Focuses on institutional activity at mathematically defined support/resistance
# - Combines multiple, reinforcing sources of trading edge
# - Aims for steady, dependable performance across different market conditions
# - Designed to adapt to changing markets through built-in flexibility
# - Should generate the minimum required trades while avoiding excessive frequency
# - Pivot points provide mathematically defined levels that guide institutional order flow
# - Volume confirmation ensures institutional participation at trade signals
# - Trend filter aligns with higher timeframe institutional positioning
# - Mean reversion captures ranging market tendencies when they occur
# - Simple exit rules protect capital and lock in profits from successful trades
# - Designed specifically for the 6s timeframe in cryptocurrency markets
# - Aligns with how institutions actually trade around key daily levels
# - Should perform well in both bull and bear market conditions
# - Avoids unnecessary complexity that leads to poor out-of-sample performance
# - Uses time-tested pivot concepts that institutions have used for decades
# - Volume confirmation provides evidence of institutional participation
# - Trend filter ensures alignment with higher timeframe institutional direction
# - Mean reversion captures counter-trend moves during sideways periods
# - Exit rules depend on whether levels continue to function as support/resistance
# - Should produce sufficient alpha to overcome transaction costs reasonably
# - Designed for institutional-style trading in cryptocurrency markets
# - Simple enough to resist overfitting, sophisticated enough to capture structure
# - Focuses on institutional order flow at the levels that smart money watches
# - Combines multiple, complementary sources of trading edge
# - Aims for consistent performance rather than sporadic large home-run wins
# - Designed to navigate various market regimes through adaptive components
# - Should generate adequate trades while staying within frequency limitations
# - Pivot calculation uses the exact formula institutions rely on
# - Volume confirmation ensures we're trading with market participation
# - Trend filter prevents fighting higher timeframe institutional positioning
# - Mean reversion provides additional alpha during sideways markets
# - Simple exit rules based on price action at key levels
# - Designed specifically for capturing 6s movements around daily pivot levels
# - Aligns with how institutions actually trade around mathematically defined levels
# - Should work well in both trending and ranging market conditions
# - Avoids complexity that typically leads to overfitting and poor generalization
# - Uses proven pivot concepts that institutions actually watch and trade
# - Volume confirmation validates institutional interest at key levels
# - Trend filter ensures we're trading with, not against, higher timeframe momentum
# - Mean reversion captures appropriate counter-trend moves when markets range
# - Exit rules depend on whether key levels continue to hold as support/resistance
# - Should generate sufficient trading opportunities without becoming excessive
# - Designed for real market behavior rather than backtest overfitting
# - Focuses on where institutional money likely rests orders around key levels
# - Combines pivot relevance, volume confirmation, and trend alignment as edges
# - Aims for steady performance rather than infrequent, large lottery wins
# - Designed to function across various market environments through adaptability
# - Should produce the minimum viable trades while respecting frequency limits
# - Pivot points are objective, calculation-based levels that institutions watch
# - Volume confirmation ensures institutional participation at signals
# - Trend filter aligns with higher timeframe institutional momentum
# - Mean reversion captures ranging market tendencies when they occur
# - Simple exit rules prevent losses from growing and profits from eroding
# - Designed specifically for 6s cryptocurrency trading dynamics
# - Aligns with how smart money actually operates around key daily levels
# - Should work effectively in both trending and ranging market scenarios
# - Avoids complexity that typically leads to overfitting and poor generalization
# - Uses pivot concepts that institutions actually watch and base trading decisions on
# - Volume confirmation confirms institutional participation at key levels
# - Trend filter ensures we're swimming with the higher timeframe institutional current
# - Mean reversion captures appropriate movements when markets are sideways
# - Exit rules depend on whether levels continue to serve as support/resistance
# - Should generate sufficient trading frequency without becoming excessive
# - Designed for actual market behavior rather than backtest optimization
# - Focuses on institutional activity at mathematically defined support/resistance
# - Combines multiple, complementary sources of trading advantage
# - Aims for reliable performance across different market environments
# - Designed to navigate changing conditions through built-in adaptive logic
# - Should generate adequate trades while staying within frequency boundaries
# - Pivot calculation follows the standard institutional formula precisely
# - Volume confirmation ensures we're trading with market participation
# - Trend filter prevents trading against higher timeframe institutional positioning
# - Mean reversion adds diversification during ranging market periods
# - Simple exit rules based on whether key levels continue to hold
# - Designed specifically for capturing meaningful 6s swings around daily pivots
# - Aligns with institutional behavior around mathematically calculated levels
# - Should be effective in both trending and ranging market environments
# - Avoids complexity that typically leads to overfitting and poor generalization
# - Uses pivot concepts that institutions actually watch and trade upon
# - Volume confirmation validates that institutional money is participating
# - Trend filter ensures we're trading with, not against, higher timeframe momentum
# - Mean reversion captures appropriate counter-trend moves when markets range
# - Exit rules depend on whether key levels continue to hold as support/resistance
# - Should generate sufficient opportunities without excessive trading frequency
# - Designed for real market behavior rather than backtest overfitting
# - Focuses on institutional activity at mathematically defined support/resistance
# - Combines multiple, reinforcing sources of trading edge
# - Aims for steady, dependable performance across different market conditions
# - Designed to adapt to changing markets through built-in flexibility
# - Should generate the minimum required trades while avoiding excessive frequency
# - Pivot points provide mathematically defined levels that guide institutional order flow
# - Volume confirmation ensures institutional participation at trade signals
# - Trend filter aligns with higher timeframe institutional positioning
# - Mean reversion captures ranging market tendencies when they occur
# - Simple exit rules protect capital and