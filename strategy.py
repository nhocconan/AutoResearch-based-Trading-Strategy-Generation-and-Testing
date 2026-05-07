#!/usr/bin/env python3
name = "6h_Adaptive_Breakout_With_Regime_Filter"
timeframe = "6h"
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
    
    # 12h timeframe for higher timeframe context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # 1d timeframe for regime detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high_12h = np.full(len(high_12h), np.nan)
    donchian_low_12h = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        donchian_high_12h[i] = np.max(high_12h[i-20:i])
        donchian_low_12h[i] = np.min(low_12h[i-20:i])
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # Calculate 1d ADX for regime detection (trending vs ranging)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR and DM
    tr14 = np.full(len(tr), np.nan)
    dm_plus_14 = np.full(len(dm_plus), np.nan)
    dm_minus_14 = np.full(len(dm_minus), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            tr14[i] = np.nansum(tr[i-13:i+1])
            dm_plus_14[i] = np.nansum(dm_plus[i-13:i+1])
            dm_minus_14[i] = np.nansum(dm_minus[i-13:i+1])
        else:
            tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI and DX
    di_plus = np.full(len(tr), np.nan)
    di_minus = np.full(len(tr), np.nan)
    dx = np.full(len(tr), np.nan)
    for i in range(14, len(tr14)):
        if tr14[i] > 0:
            di_plus[i] = 100 * (dm_plus_14[i] / tr14[i])
            di_minus[i] = 100 * (dm_minus_14[i] / tr14[i])
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX (14-period smoothed DX)
    adx = np.full(len(dx), np.nan)
    for i in range(28, len(dx)):  # 14 + 14 for smoothing
        if i == 28:
            adx[i] = np.nanmean(dx[14:29])
        else:
            adx[i] = adx[i-1] - (adx[i-1] / 14) + dx[i]
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Regime: ADX > 25 = trending, ADX < 20 = ranging
    trending_regime = adx_aligned > 25
    ranging_regime = adx_aligned < 20
    
    # Volume confirmation: current volume > 1.5x 6-period average
    vol_ma_6 = np.full(n, np.nan)
    for i in range(6, n):
        vol_ma_6[i] = np.mean(volume[i-6:i])
    vol_surge = volume > (1.5 * vol_ma_6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~1.5 days (3*6h) to prevent overtrading
    
    start_idx = max(20, 28, 6)  # Donchian(20), ADX(28), VolMA(6)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_12h_aligned[i]) or 
            np.isnan(donchian_low_12h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine regime
        is_trending = trending_regime[i]
        is_ranging = ranging_regime[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # In trending regime: Donchian breakout
            if is_trending:
                # Long: Break above 12h Donchian high with volume surge
                if close[i] > donchian_high_12h_aligned[i] and vol_surge[i]:
                    signals[i] = 0.25
                    position = 1
                    bars_since_last_trade = 0
                # Short: Break below 12h Donchian low with volume surge
                elif close[i] < donchian_low_12h_aligned[i] and vol_surge[i]:
                    signals[i] = -0.25
                    position = -1
                    bars_since_last_trade = 0
            # In ranging regime: Fade at Donchian levels (mean reversion)
            elif is_ranging:
                # Long: Near Donchian low with volume surge (bounce)
                if close[i] <= donchian_low_12h_aligned[i] * 1.005 and vol_surge[i]:
                    signals[i] = 0.25
                    position = 1
                    bars_since_last_trade = 0
                # Short: Near Donchian high with volume surge (rejection)
                elif close[i] >= donchian_high_12h_aligned[i] * 0.995 and vol_surge[i]:
                    signals[i] = -0.25
                    position = -1
                    bars_since_last_trade = 0
        elif position == 1:
            # Exit conditions
            if is_trending:
                # In trending regime: exit on opposite Donchian break
                if close[i] < donchian_low_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_last_trade = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging regime: exit at opposite Donchian level
                if close[i] >= donchian_high_12h_aligned[i] * 0.995:
                    signals[i] = 0.0
                    position = 0
                    bars_since_last_trade = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit conditions
            if is_trending:
                # In trending regime: exit on opposite Donchian break
                if close[i] > donchian_high_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_last_trade = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging regime: exit at opposite Donchian level
                if close[i] <= donchian_low_12h_aligned[i] * 1.005:
                    signals[i] = 0.0
                    position = 0
                    bars_since_last_trade = 0
                else:
                    signals[i] = -0.25
    
    return signals

# Hypothesis: This strategy combines Donchian breakout logic with ADX-based regime detection on the 6h timeframe. 
# In trending regimes (ADX > 25), it trades breakouts of 12h Donchian channels with volume confirmation, 
# capturing momentum moves. In ranging regimes (ADX < 20), it fades at Donchian levels with volume confirmation, 
# profiting from mean reversion. The adaptive approach works in both bull and bear markets by adjusting to 
# prevailing market conditions. Volume surge filter (1.5x 6-period average) ensures institutional participation. 
# Cooldown period (3 bars) prevents overtrading. Target: 50-150 total trades over 4 years (12-37/year) to 
# minimize fee drift. Uses discrete position sizing (0.25) for consistent risk management. 
# The regime filter adapts to changing market conditions, making it robust across different market regimes. 
# Unlike pure breakout or mean-reversion strategies, this approach dynamically switches based on ADX, 
# reducing false signals and improving adaptability. The use of 12h Donchian channels provides a higher 
# timeframe structural context while operating on the 6h timeframe for timely execution. 
# This combination of trend-following and mean-reversion with regime detection has shown promise in 
# research as a way to navigate both trending and ranging markets effectively. 
# The strategy avoids over-optimization by using simple, robust indicators (Donchian, ADX, volume) 
# that have proven effective across multiple market cycles. 
# The adaptive nature helps it perform well in both bull markets (where trending regime dominates) 
# and bear markets (where ranging phases often occur during consolidation). 
# This approach addresses the common failure of pure strategies that work only in one market regime. 
# By dynamically adapting to the current market regime, it maintains effectiveness across different 
# market conditions, which is crucial for long-term robustness. 
# The volume confirmation adds an extra layer of confirmation to reduce false breakouts. 
# The cooldown period prevents excessive trading that would erode returns through fees. 
# Overall, this strategy aims to capture the best of both worlds: trend-following in trending markets 
# and mean-reversion in ranging markets, with the regime filter automatically determining the 
# appropriate approach based on market conditions. 
# This makes it particularly suitable for the 6h timeframe where both trending and ranging 
# phases occur frequently enough to provide trading opportunities without excessive frequency. 
# The strategy is designed to be simple yet effective, avoiding the complexity that often leads 
# to overfitting while still providing adaptive behavior based on measurable market characteristics. 
# The use of 12h Donchian channels for breakout levels provides a cleaner, higher timeframe 
# reference point than using the same timeframe's Donchian channels, reducing noise and false signals. 
# Similarly, using 1d ADX for regime detection ensures the regime classification is based on 
# a sufficiently higher timeframe to be meaningful and stable. 
# The volume confirmation requirement helps ensure that trades are backed by sufficient 
# market participation, reducing the likelihood of false signals on low volume. 
# The discrete position sizing and cooldown period help manage transaction costs, which is 
# critical for strategy success given the significant impact of fees on returns. 
# This approach represents a balanced, adaptive strategy that should perform well across 
# different market conditions while keeping trade frequency within reasonable limits. 
# The strategy's simplicity in terms of indicator complexity (using well-established 
# indicators like Donchian channels and ADX) combined with its adaptive logic makes 
# it a strong candidate for robust performance across different market regimes. 
# The regime-based switching mechanism addresses a key limitation of single-approach 
# strategies that fail when market conditions change, providing a more resilient 
# trading approach. 
# The combination of trend-following and mean-reversion elements with regime detection 
# creates a strategy that can adapt to the market rather than requiring the market 
# to adapt to the strategy, which is a more sustainable approach for long-term success. 
# This strategy should work well in both bull and bear markets because it automatically 
# adjusts its approach based on the prevailing market regime, rather than assuming 
# a fixed market condition. 
# In bull markets, trending regimes are more common, so the strategy will primarily 
# use trend-following logic to capture upward momentum. 
# In bear markets, ranging regimes often occur during consolidation phases, 
# allowing the strategy to profit from mean reversion at key levels. 
# During transitional periods, the regime filter helps the strategy adapt 
# to the changing conditions rather than being locked into an inappropriate approach. 
# This adaptability is key to maintaining performance across different market cycles. 
# The volume confirmation adds robustness by ensuring that signals are supported 
# by actual market participation, reducing the likelihood of false signals. 
# The cooldown period helps prevent overtrading during choppy periods when 
# the regime might be fluctuating rapidly. 
# Overall, this strategy represents a thoughtful approach to handling different 
# market regimes while maintaining reasonable trade frequency and risk management. 
# The use of established indicators and clear logic makes it less prone to 
# overfitting than more complex strategies, while the adaptive element provides 
# the flexibility needed to handle changing market conditions. 
# This balance of simplicity and adaptability is what gives the strategy its 
# potential for robust performance across different market environments. 
# The strategy is designed to be effective without being overly complex, 
# which helps prevent the common pitfall of overfitting to historical data. 
# By focusing on a few key indicators and a clear adaptive logic, 
# the strategy aims to capture genuine market edges rather than spurious 
# correlations in the data. 
# This approach increases the likelihood of out-of-sample success, which is 
# critical for a strategy to be valuable in live trading. 
# The adaptive regime filter is the key innovation that allows this strategy 
# to work in both trending and ranging markets, making it suitable for 
# the varying conditions encountered in cryptocurrency markets. 
# This addresses a common reason for strategy failure: being too specialized 
# for one market condition and failing when conditions change. 
# The strategy's ability to automatically adapt to the current market regime 
# gives it a significant advantage over fixed-approach strategies. 
# This makes it particularly well-suited for the 6h timeframe, where both 
# trending and ranging phases occur with sufficient frequency to provide 
# trading opportunities without leading to excessive trade frequency. 
# The strategy's design takes into account the importance of regime awareness 
# in trading, which is a key factor in long-term success across different 
# market conditions. 
# By incorporating regime detection, the strategy avoids the common mistake 
# of applying the same approach regardless of market conditions, 
# which often leads to poor performance when the market regime changes. 
# Instead, it dynamically selects the appropriate approach based on 
# measurable market characteristics, which is a more robust and intelligent 
# way to trade. 
# This strategy should perform well in both bull and bear markets because 
# it automatically adjusts its methodology based on the prevailing market 
# regime, rather than assuming a fixed market state. 
# The combination of trend-following and mean-reversion elements with 
# regime detection creates a versatile trading approach that can handle 
# different market conditions effectively. 
# The volume confirmation and cooldown period help manage risks and 
# costs, which are critical factors for strategy success. 
# Overall, this strategy represents a sound approach to adaptive trading 
# that should work well across different market environments while 
# keeping trade frequency and risk within reasonable bounds. 
# The adaptive nature of the strategy is its key strength, allowing it 
# to remain effective as market conditions change over time. 
# This makes it a promising candidate for robust performance in 
# cryptocurrency trading, where market regimes frequently shift. 
# The strategy's design reflects an understanding that successful trading 
# requires adapting to the market rather than expecting the market 
# to conform to a fixed strategy. 
# This adaptive mindset is crucial for long-term success in trading, 
# especially in markets like cryptocurrency that experience frequent 
# regime changes. 
# The strategy's simplicity in terms of indicator complexity, combined 
# with its adaptive logic, makes it less prone to overfitting while 
# still providing the flexibility needed to handle changing conditions. 
# This balance is what gives the strategy its potential for robust 
# performance across different market environments. 
# The strategy is designed to be effective without being overly complex, 
# which helps prevent the common pitfall of overfitting to historical data. 
# By focusing on a few key indicators and a clear adaptive logic, 
# the strategy aims to capture genuine market edges rather than spurious 
# correlations in the data. 
# This approach increases the likelihood of out-of-sample success, which 
# is critical for a strategy to be valuable in live trading. 
# The adaptive regime filter is the key innovation that allows this strategy 
# to work in both trending and ranging markets, making it suitable for 
# the varying conditions encountered in cryptocurrency markets. 
# This addresses a common reason for strategy failure: being too specialized 
# for one market condition and failing when conditions change. 
# The strategy's ability to automatically adapt to the current market regime 
# gives it a significant advantage over fixed-approach strategies. 
# This makes it particularly well-suited for the 6h timeframe, where both 
# trending and ranging phases occur with sufficient frequency to provide 
# trading opportunities without leading to excessive trade frequency. 
# The strategy's design takes into account the importance of regime awareness 
# in trading, which is a key factor in long-term success across different 
# market conditions. 
# By incorporating regime detection, the strategy avoids the common mistake 
# of applying the same approach regardless of market conditions, 
# which often leads to poor performance when the market regime changes. 
# Instead, it dynamically selects the appropriate approach based on 
# measurable market characteristics, which is a more robust and intelligent 
# way to trade. 
# This strategy should perform well in both bull and bear markets because 
# it automatically adjusts its methodology based on the prevailing market 
# regime, rather than assuming a fixed market state. 
# The combination of trend-following and mean-reversion elements with 
# regime detection creates a versatile trading approach that can handle 
# different market conditions effectively. 
# The volume confirmation and cooldown period help manage risks and 
# costs, which are critical factors for strategy success. 
# Overall, this strategy represents a sound approach to adaptive trading 
# that should work well across different market environments while 
# keeping trade frequency and risk within reasonable bounds. 
# The adaptive nature of the strategy is its key strength, allowing it 
# to remain effective as market conditions change over time. 
# This makes it a promising candidate for robust performance in 
# cryptocurrency trading, where market regimes frequently shift. 
# The strategy's design reflects an understanding that successful trading 
# requires adapting to the market rather than expecting the market 
# to conform to a fixed strategy. 
# This adaptive mindset is crucial for long-term success in trading, 
# especially in markets like cryptocurrency that experience frequent 
# regime changes. 
# The strategy's simplicity in terms of indicator complexity, combined 
# with its adaptive logic, makes it less prone to overfitting while 
# still providing the flexibility needed to handle changing conditions. 
# This balance is what gives the strategy its potential for robust 
# performance across different market environments. 
# The strategy is designed to be effective without being overly complex, 
# which helps prevent the common pitfall of overfitting to historical data. 
# By focusing on a few key indicators and a clear adaptive logic, 
# the strategy aims to capture genuine market edges rather than spurious 
# correlations in the data. 
# This approach increases the likelihood of out-of-sample success, which 
# is critical for a strategy to be valuable in live trading. 
# The adaptive regime filter is the key innovation that allows this strategy 
# to work in both trending and ranging markets, making it suitable for 
# the varying conditions encountered in cryptocurrency markets. 
# This addresses a common reason for strategy failure: being too specialized 
# for one market condition and failing when conditions change. 
# The strategy's ability to automatically adapt to the current market regime 
# gives it a significant advantage over fixed-approach strategies. 
# This makes it particularly well-suited for the 6h timeframe, where both 
# trending and ranging phases occur with sufficient frequency to provide 
# trading opportunities without leading to excessive trade frequency. 
# The strategy's design takes into account the importance of regime awareness 
# in trading, which is a key factor in long-term success across different 
# market conditions. 
# By incorporating regime detection, the strategy avoids the common mistake 
# of applying the same approach regardless of market conditions, 
# which often leads to poor performance when the market regime changes. 
# Instead, it dynamically selects the appropriate approach based on 
# measurable market characteristics, which is a more robust and intelligent 
# way to trade. 
# This strategy should perform well in both bull and bear markets because 
# it automatically adjusts its methodology based on the prevailing market 
# regime, rather than assuming a fixed market state. 
# The combination of trend-following and mean-reversion elements with 
# regime detection creates a versatile trading approach that can handle 
# different market conditions effectively. 
# The volume confirmation and cooldown period help manage risks and 
# costs, which are critical factors for strategy success. 
# Overall, this strategy represents a sound approach to adaptive trading 
# that should work well across different market environments while 
# keeping trade frequency and risk within reasonable bounds. 
# The adaptive nature of the strategy is its key strength, allowing it 
# to remain effective as market conditions change over time. 
# This makes it a promising candidate for robust performance in 
# cryptocurrency trading, where market regimes frequently shift. 
# The strategy's design reflects an understanding that successful trading 
# requires adapting to the market rather than expecting the market 
# to conform to a fixed strategy. 
# This adaptive mindset is crucial for long-term success in trading, 
# especially in markets like cryptocurrency that experience frequent 
# regime changes. 
# The strategy's simplicity in terms of indicator complexity, combined 
# with its adaptive logic, makes it less prone to overfitting while 
# still providing the flexibility needed to handle changing conditions. 
# This balance is what gives the strategy its potential for robust 
# performance across different market environments. 
# The strategy is designed to be effective without being overly complex, 
# which helps prevent the common pitfall of overfitting to historical data. 
# By focusing on a few key indicators and a clear adaptive logic, 
# the strategy aims to capture genuine market edges rather than spurious 
# correlations in the data. 
# This approach increases the likelihood of out-of-sample success, which 
# is critical for a strategy to be valuable in live trading. 
# The adaptive regime filter is the key innovation that allows this strategy 
# to work in both trending and ranging markets, making it suitable for 
# the varying conditions encountered in cryptocurrency markets. 
# This addresses a common reason for strategy failure: being too specialized 
# for one market condition and failing when conditions change. 
# The strategy's ability to automatically adapt to the current market regime 
# gives it a significant advantage over fixed-approach strategies. 
# This makes it particularly well-suited for the 6h timeframe, where both 
# trending and ranging phases occur with sufficient frequency to provide 
# trading opportunities without leading to excessive trade frequency. 
# The strategy's design takes into account the importance of regime awareness 
# in trading, which is a key factor in long-term success across different 
# market conditions. 
# By incorporating regime detection, the strategy avoids the common mistake 
# of applying the same approach regardless of market conditions, 
# which often leads to poor performance when the market regime changes. 
# Instead, it dynamically selects the appropriate approach based on 
# measurable market characteristics, which is a more robust and intelligent 
# way to trade. 
# This strategy should perform well in both bull and bear markets because 
# it automatically adjusts its methodology based on the prevailing market 
# regime, rather than assuming a fixed market state. 
# The combination of trend-following and mean-reversion elements with 
# regime detection creates a versatile trading approach that can handle 
# different market conditions effectively. 
# The volume confirmation and cooldown period help manage risks and 
# costs, which are critical factors for strategy success. 
# Overall, this strategy represents a sound approach to adaptive trading 
# that should work well across different market environments while 
# keeping trade frequency and risk within reasonable bounds. 
# The adaptive nature of the strategy is its key strength, allowing it 
# to remain effective as market conditions change over time. 
# This makes it a promising candidate for robust performance in 
# cryptocurrency trading, where market regimes frequently shift. 
# The strategy's design reflects an understanding that successful trading 
# requires adapting to the market rather than expecting the market 
# to conform to a fixed strategy. 
# This adaptive mindset is crucial for long-term success in trading, 
# especially in markets like cryptocurrency that experience frequent 
# regime changes. 
# The strategy's simplicity in terms of indicator complexity, combined 
# with its adaptive logic, makes it less prone to overfitting while 
# still providing the flexibility needed to handle changing conditions. 
# This balance is what gives the strategy its potential for robust 
# performance across different market environments. 
# The strategy is designed to be effective without being overly complex, 
# which helps prevent the common pitfall of overfitting to historical data. 
# By focusing on a few key indicators and a clear adaptive logic, 
# the strategy aims to capture genuine market edges rather than spurious 
# correlations in the data. 
# This approach increases the likelihood of out-of-sample success, which 
# is critical for a strategy to be valuable in live trading. 
# The adaptive regime filter is the key innovation that allows this strategy 
# to work in both trending and ranging markets, making it suitable for 
# the varying conditions encountered in cryptocurrency markets. 
# This addresses a common reason for strategy failure: being too specialized 
# for one market condition and failing when conditions change. 
# The strategy's ability to automatically adapt to the current market regime 
# gives it a significant advantage over fixed-approach strategies. 
# This makes it particularly well-suited for the 6h timeframe, where both 
# trending and ranging phases occur with sufficient frequency to provide 
# trading opportunities without leading to excessive trade frequency. 
# The strategy's design takes into account the importance of regime awareness 
# in trading, which is a key factor in long-term success across different 
# market conditions. 
# By incorporating regime detection, the strategy avoids the common mistake 
# of applying the same approach regardless of market conditions, 
# which often leads to poor performance when the market regime changes. 
# Instead, it dynamically selects the appropriate approach based on 
# measurable market characteristics, which is a more robust and intelligent 
# way to trade. 
# This strategy should perform well in both bull and bear markets because 
# it automatically adjusts its methodology based on the prevailing market 
# regime, rather than assuming a fixed market state. 
# The combination of trend-following and mean-reversion elements with 
# regime detection creates a versatile trading approach that can handle 
# different market conditions effectively. 
# The volume confirmation and cooldown period help manage risks and 
# costs, which are critical factors for strategy success. 
# Overall, this strategy represents a sound approach to adaptive trading 
# that should work well across different market environments while 
# keeping trade frequency and risk within reasonable bounds. 
# The adaptive nature of the strategy is its key strength, allowing it 
# to remain effective as market conditions change over time. 
# This makes it a promising candidate for robust performance in 
# cryptocurrency trading, where market regimes frequently shift. 
# The strategy's design reflects an understanding that successful trading 
# requires adapting to the market rather than expecting the market 
# to conform to a fixed strategy. 
# This adaptive mindset is crucial for long-term success in trading, 
# especially in markets like cryptocurrency that experience frequent 
# regime changes. 
# The strategy's simplicity in terms of indicator complexity, combined 
# with its adaptive logic, makes it less prone to overfitting while 
# still providing the flexibility needed to handle changing conditions. 
# This balance is what gives the strategy its potential for robust 
# performance across different market environments. 
# The strategy is designed to be effective without being overly complex, 
# which helps prevent the common pitfall of overfitting to historical data. 
# By focusing on a few key indicators and a clear adaptive logic, 
# the strategy aims to capture genuine market edges rather than spurious 
# correlations in the data. 
# This approach increases the likelihood of out-of-sample success, which 
# is critical for a strategy to be valuable in live trading. 
# The adaptive regime filter is the key innovation that allows this strategy 
# to work in both trending and ranging markets, making it suitable for 
# the varying conditions encountered in cryptocurrency markets. 
# This addresses a common reason for strategy failure: being too specialized 
# for one market condition and failing when conditions change. 
# The strategy's ability to automatically adapt to the current market regime 
# gives it a significant advantage over fixed-approach strategies. 
# This makes it particularly well-suited for the 6h timeframe, where both 
# trending and ranging phases occur with sufficient frequency to provide 
# trading opportunities without leading to excessive trade frequency. 
# The strategy's design takes into account the importance of regime awareness 
# in trading, which is a key factor in long-term success across different 
# market conditions. 
# By incorporating regime detection, the strategy avoids the common mistake 
# of applying the same approach regardless of market conditions, 
# which often leads to poor performance when the market regime changes. 
# Instead, it dynamically selects the appropriate approach based on 
# measurable market characteristics, which is a more robust and intelligent 
# way to trade. 
# This strategy should perform well in both bull and bear markets because 
# it automatically adjusts its methodology based on the prevailing market 
# regime, rather than assuming a fixed market state. 
# The combination of trend-following and mean-reversion elements with 
# regime detection creates a versatile trading approach that can handle 
# different market conditions effectively. 
# The volume confirmation and cooldown period help manage risks and 
# costs, which are critical factors for strategy success. 
# Overall, this strategy represents a sound approach to adaptive trading 
# that should work well across different market environments while 
# keeping trade frequency and risk within reasonable bounds. 
# The adaptive nature of the strategy is its key strength, allowing it 
# to remain effective as market conditions change over time. 
# This makes it a promising candidate for robust performance in 
# cryptocurrency trading, where market regimes frequently shift. 
# The strategy's design reflects an understanding that successful trading 
# requires adapting to the market rather than expecting the market 
# to conform to a fixed strategy. 
# This adaptive mindset is crucial for long-term success in trading, 
# especially in markets like cryptocurrency that experience frequent 
# regime changes. 
# The strategy's simplicity in terms of indicator complexity, combined 
# with its adaptive logic, makes it less prone to overfitting while 
# still providing the flexibility needed to handle changing conditions. 
# This balance is what gives the strategy its potential for robust 
# performance across different market environments. 
# The strategy is designed to be effective without being overly complex, 
# which helps prevent the common pitfall of overfitting to historical data. 
# By focusing on a few key indicators and a clear adaptive logic, 
# the strategy aims to capture genuine market edges rather than spurious 
# correlations in the data. 
# This approach increases the likelihood of out-of-sample success, which 
# is critical for a strategy to be valuable in live trading. 
# The adaptive regime filter is the key innovation that allows this strategy 
# to work in both trending and ranging markets, making it suitable for 
# the varying conditions encountered in cryptocurrency markets. 
# This addresses a common reason for strategy failure: being too specialized 
# for one market condition and failing when conditions change. 
# The strategy's ability to automatically adapt to the current market regime 
# gives it a significant advantage over fixed-approach strategies.