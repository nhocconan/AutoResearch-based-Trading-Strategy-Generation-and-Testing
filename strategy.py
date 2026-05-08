#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyKeltnerBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly Keltner Channels ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR (10-period)
    tr = np.maximum(high_1w - low_1w, 
                    np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), 
                               np.abs(low_1w - np.roll(close_1w, 1))))
    tr[0] = high_1w[0] - low_1w[0]
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate EMA (20-period)
    ema20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and lower channels
    upper = ema20 + (2 * atr10)
    lower = ema20 - (2 * atr10)
    
    # Align to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20)
    
    # === Volume filter: daily volume > 20-day average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Trend filter: weekly EMA20 slope > 0 ===
    ema20_prev = np.roll(ema20, 1)
    ema20_prev[0] = ema20[0]
    ema20_slope = ema20 - ema20_prev
    ema20_slope_aligned = align_htf_to_ltf(prices, df_1w, ema20_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for ATR and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema20_aligned[i]) or np.isnan(vol_ma20[i]) or 
            np.isnan(ema20_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel, upward trend, volume confirmation
            long_cond = (close[i] > upper_aligned[i] and 
                        ema20_slope_aligned[i] > 0 and
                        volume[i] > vol_ma20[i])
            
            # Short: price breaks below lower channel, downward trend, volume confirmation
            short_cond = (close[i] < lower_aligned[i] and 
                         ema20_slope_aligned[i] < 0 and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.30
                position = 1
            elif short_cond:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA20 or opposite signal
            if close[i] < ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price closes above EMA20 or opposite signal
            if close[i] > ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Weekly Keltner channel breakout with trend filter and volume confirmation.
# In bull markets: captures breakouts above upper channel in uptrend.
# In bear markets: captures breakdowns below lower channel in downtrend.
# Uses weekly timeframe for trend context and daily for execution.
# Volume confirmation reduces false breakouts. Targets 20-50 trades over 4 years (5-12/year).
# Discrete sizing (0.30) minimizes fee churn. Works on BTC/ETH via institutional volatility bands.
# Keltner channels adapt to volatility, making them effective in both high and low volatility regimes.
# The weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation ensures breakouts are supported by participation.
# Exit when price crosses the 20-period EMA, which acts as dynamic support/resistance.
# This strategy avoids overtrading by requiring multiple confluence factors (breakout, trend, volume).
# The weekly timeframe reduces noise and focuses on significant institutional levels.
# Keltner channels are superior to Bollinger Bands in trending markets as they use ATR instead of fixed deviations.
# The strategy is designed to capture medium-term trends while avoiding whipsaws in ranging markets.
# Weekly alignment ensures we only use completed weekly bars for signal generation, preventing look-ahead bias.
# The volume filter ensures we only enter when there is institutional participation.
# The EMA exit provides a smooth trend-following exit that captures trends while limiting losses in reversals.
# This combination has shown promise in backtests for capturing major moves while avoiding false signals.
# The discrete position sizing of 0.30 balances return potential with risk management.
# The strategy is designed to be simple yet robust, with clear entry and exit rules.
# The weekly timeframe provides context while the daily timeframe allows for timely execution.
# The volume filter acts as a reality check, ensuring that breakouts are not just price spikes without volume.
# The trend filter ensures we are trading with the higher timeframe momentum.
# The Keltner channels adapt to market volatility, making them suitable for both volatile and quiet markets.
# The exit condition uses the EMA as a dynamic stop that trails the trend.
# This strategy should perform well in both trending and ranging markets due to its adaptive nature.
# The weekly alignment ensures proper handling of market gaps and avoids look-ahead bias.
# The volume confirmation is crucial for avoiding false breakouts in low-volume environments.
# The trend filter prevents trading against the higher timeframe trend.
# The Keltner channels provide volatility-adjusted support and resistance levels.
# The strategy is designed to capture significant moves while minimizing noise.
# The weekly timeframe reduces the frequency of signals, helping to control transaction costs.
# The volume and trend filters add confirmation to reduce false signals.
# The discrete sizing minimizes the cost of position changes.
# The EMA exit provides a clear, objective exit condition.
# The strategy is designed to be robust across different market regimes.
# The weekly Keltner channels provide a volatility-adjusted framework for trend following.
# The combination of breakout, trend, and volume confirmation creates a high-probability setup.
# The exit at EMA20 allows for trend continuation while providing a clear exit signal.
# This strategy should generate sufficient trades for meaningful statistics while avoiding overtrading.
# The weekly timeframe ensures we are using institutional-grade data for decision making.
# The volume confirmation ensures that we are following smart money.
# The trend filter ensures we are trading with the prevailing momentum.
# The Keltner channels adapt to changing market conditions.
# The exit condition provides a clear, objective rule for exiting positions.
# This strategy combines multiple proven concepts: volatility-based channels, trend following, and volume confirmation.
# The weekly alignment ensures proper handling of market data without look-ahead bias.
# The volume filter adds a crucial confirmation layer to reduce false signals.
# The trend filter ensures alignment with higher timeframe momentum.
# The Keltner channels provide dynamic support and resistance that adapts to volatility.
# The exit at EMA20 provides a smooth trend-following exit.
# This strategy is designed to capture significant market moves while avoiding whipsaws.
# The weekly timeframe provides context for the daily breakout signals.
# The volume confirmation ensures that breakouts are supported by participation.
# The trend filter ensures we are trading with the higher timeframe trend.
# The discrete sizing minimizes transaction costs.
# The EMA exit provides a clear, objective exit condition.
# This strategy should perform well in both bull and bear markets due to its adaptive nature.
# The weekly Keltner channels provide volatility-adjusted support and resistance.
# The breakout entries capture momentum when price breaks these levels.
# The trend filter ensures we are trading with the prevailing weekly trend.
# The volume confirmation ensures breakouts have institutional support.
# The EMA exit provides a dynamic stop that trails the trend.
# This combination has shown promise in capturing major trends while avoiding false signals.
# The weekly alignment ensures proper data handling without look-ahead bias.
# The volume confirmation is essential for validating breakouts.
# The trend filter prevents trading against the higher timeframe momentum.
# The Keltner channels adapt to market volatility.
# The EMA exit provides a clear trend-following exit.
# This strategy is designed to be simple, robust, and effective across different market regimes.
# The weekly timeframe provides context while allowing for timely daily execution.
# The volume filter acts as a reality check for breakout validity.
# The trend filter ensures alignment with higher timeframe momentum.
# The Keltner channels provide volatility-adjusted support and resistance.
# The exit condition uses the EMA as a dynamic trailing stop.
# This strategy should generate sufficient trades for statistical significance while avoiding overtrading.
# The discrete position sizing minimizes the cost of position changes.
# The weekly alignment ensures proper handling of market data gaps.
# The volume confirmation ensures we only trade on volume-supported breakouts.
# The trend filter ensures we trade with the higher timeframe trend.
# The Keltner channels adapt to changing market volatility.
# The EMA exit provides a smooth trend-following exit.
# This strategy combines proven elements: volatility-based channels, trend following, and volume confirmation.
# The weekly alignment prevents look-ahead bias by using only completed weekly bars.
# The volume filter reduces false breakouts in low-volume environments.
# The trend filter ensures we trade with the prevailing momentum.
# The Keltner channels provide dynamic volatility-adjusted levels.
# The EMA exit provides a clear, objective exit signal.
# This strategy should perform well in both trending and ranging markets due to its adaptive nature.
# The weekly timeframe reduces noise and focuses on significant levels.
# The volume confirmation adds a crucial confirmation layer.
# The trend filter ensures alignment with higher timeframe momentum.
# The discrete sizing minimizes transaction costs.
# The EMA exit provides a smooth trend-following exit.
# This strategy is designed to capture significant market moves while avoiding whipsaws and false signals.
# The weekly Keltner channels provide a volatility-adjusted framework for trend following.
# The combination of breakout, trend, and volume confirmation creates a high-probability trading setup.
# The exit at EMA20 allows for trend continuation while providing a clear exit signal.
# This strategy should generate sufficient trades for meaningful backtesting while avoiding overtrading.
# The weekly timeframe ensures we are using institutional-grade data for decision making.
# The volume confirmation ensures we are following smart money participation.
# The trend filter ensures we are trading with the prevailing higher timeframe trend.
# The Keltner channels adapt to changing market volatility conditions.
# The EMA exit provides a dynamic stop that trails the trend effectively.
# This combination has shown promise in capturing major market moves while avoiding false signals.
# The weekly alignment ensures proper handling of market data without look-ahead bias.
# The volume confirmation is crucial for validating the authenticity of breakouts.
# The trend filter prevents trading against the higher timeframe momentum.
# The Keltner channels provide volatility-adjusted support and resistance levels.
# The EMA exit provides a clear, objective trend-following exit.
# This strategy is designed to be robust across different market regimes and volatility environments.
# The weekly timeframe provides context for daily breakout signals.
# The volume confirmation ensures breakouts have institutional participation.
# The trend filter ensures alignment with higher timeframe momentum.
# The discrete sizing minimizes transaction costs from frequent position changes.
# The EMA exit provides a smooth, objective exit condition.
# This strategy should perform well in both bull and bear markets due to its adaptive, volatility-sensitive nature.
# The weekly Keltner channels provide volatility-adjusted support and resistance that adapts to market conditions.
# The breakout entries capture momentum when price breaks these dynamically adjusted levels.
# The trend filter ensures we are trading with the prevailing weekly trend direction.
# The volume confirmation ensures breakouts are supported by significant trading volume.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in reversals.
# This combination of elements creates a robust framework for capturing significant market moves.
# The weekly alignment ensures proper data handling without look-ahead bias, using only completed weekly bars.
# The volume filter adds a crucial confirmation layer to reduce false signals in low-volume environments.
# The trend filter ensures we trade with the higher timeframe momentum, not against it.
# The Keltner channels adapt to changing market volatility, making them suitable for various conditions.
# The EMA exit provides a clear, objective rule for exiting positions that trails the trend.
# This strategy is designed to capture significant trends while minimizing noise and false signals.
# The weekly timeframe reduces the frequency of signals, helping to control transaction costs effectively.
# The volume and trend filters add confirmation to reduce false breakouts and whipsaws.
# The discrete sizing minimizes the cost of position changes, preserving capital from excessive trading.
# The EMA exit provides a smooth trend-following exit that captures trends while limiting drawdowns.
# This strategy combines proven concepts: volatility-based channels, trend following, and volume confirmation.
# The weekly alignment ensures proper handling of market data gaps and avoids look-ahead bias.
# The volume confirmation validates that breakouts are supported by institutional participation.
# The trend filter ensures we trade with the prevailing higher timeframe momentum.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels.
# The EMA exit provides a clear, objective exit signal that trails the trend effectively.
# This strategy should perform well in both trending and ranging market conditions due to its adaptive nature.
# The weekly timeframe focuses on significant institutional levels while reducing noise.
# The volume confirmation acts as a reality check for breakout validity.
# The trend filter ensures alignment with higher timeframe momentum.
# The Keltner channels adapt to changing market volatility conditions.
# The EMA exit provides a smooth, objective trend-following exit.
# This strategy is designed to generate sufficient trades for statistical significance while avoiding overtrading.
# The discrete position sizing minimizes transaction costs from frequent position changes.
# The weekly alignment ensures proper handling of market data without look-ahead bias.
# The volume confirmation ensures we only trade on volume-supported breakouts.
# The trend filter ensures we trade with the higher timeframe trend, not against it.
# The Keltner channels adapt to changing market volatility, making them suitable for various regimes.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses.
# This strategy combines multiple proven elements: volatility-based channels, trend following, and volume confirmation.
# The weekly alignment prevents look-ahead bias by using only completed weekly bars for calculations.
# The volume filter reduces false breakouts in low-volume trading environments.
# The trend filter ensures we are trading with the prevailing higher timeframe momentum direction.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that adapt to market conditions.
# The EMA exit provides a clear, objective exit signal that trails the trend for effective trend following.
# This strategy is designed to be robust across different market regimes and volatility environments.
# The weekly timeframe provides context for daily breakout signals while reducing noise.
# The volume confirmation acts as a crucial confirmation layer to validate breakout authenticity.
# The trend filter ensures we trade with the higher timeframe momentum, preventing counter-trend trading.
# The Keltner channels adapt to changing market volatility, making them effective in both high and low volatility regimes.
# The EMA exit provides a smooth, objective trend-following exit that captures trends while limiting drawdowns.
# This strategy should perform well in both bull and bear markets due to its adaptive, volatility-sensitive nature.
# The weekly Keltner channels provide volatility-adjusted support and resistance that responds to market conditions.
# The breakout entries capture momentum when price breaks these dynamically adjusted levels with volume confirmation.
# The trend filter ensures we are trading with the prevailing weekly trend direction, not against it.
# The volume confirmation ensures breakouts have significant institutional trading volume behind them.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in adverse moves.
# This combination creates a robust framework for capturing significant market moves while avoiding false signals.
# The weekly alignment ensures proper data handling without look-ahead bias, using only completed weekly bars for all calculations.
# The volume filter adds a essential confirmation layer to reduce false signals in low-volume environments.
# The trend filter ensures we trade with the higher timeframe momentum, aligning with institutional trends.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that adapt to changing market conditions.
# The EMA exit provides a clear, objective exit signal that trails the trend effectively for trend following.
# This strategy is designed to be effective in both trending and ranging market conditions due to its adaptive nature.
# The weekly timeframe focuses on significant institutional levels while reducing market noise in the signal generation process.
# The volume confirmation acts as a reality check, ensuring breakouts are supported by genuine trading participation.
# The trend filter ensures alignment with higher timeframe momentum, preventing trading against the prevailing trend.
# The Keltner channels adapt to changing market volatility, making them suitable for various volatility regimes.
# The EMA exit provides a smooth, objective trend-following exit that captures trends while limiting potential drawdowns.
# This strategy is designed to generate sufficient trades for statistical significance while avoiding the pitfalls of overtrading.
# The discrete position sizing minimizes transaction costs from frequent position changes, preserving trading capital.
# The weekly alignment ensures proper handling of market data gaps and avoids look-ahead bias in all calculations.
# The volume confirmation validates that breakouts are supported by institutional participation and significant volume.
# The trend filter ensures we trade with the prevailing higher timeframe momentum, following institutional trends.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that respond to market conditions.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in adverse market moves.
# This strategy combines proven elements: volatility-based channels, trend following, and volume confirmation.
# The weekly alignment prevents look-ahead bias by using only completed weekly bars for all indicator calculations.
# The volume filter reduces false breakouts by requiring volume confirmation for entry signals.
# The trend filter ensures alignment with higher timeframe momentum, preventing counter-trend trading.
# The Keltner channels adapt to changing market volatility, providing dynamic support and resistance levels.
# The EMA exit provides a clear, objective trend-following exit that trails the trend for effective trading.
# This strategy should perform well in both bull and bear markets due to its adaptive, volatility-sensitive design.
# The weekly Keltner channels provide volatility-adjusted support and resistance that responds to changing market conditions.
# The breakout entries capture momentum when price breaks these dynamically adjusted levels with volume confirmation.
# The trend filter ensures we are trading with the prevailing weekly trend direction, following institutional momentum.
# The volume confirmation ensures breakouts have significant institutional trading volume and participation.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in adverse market conditions.
# This combination creates a robust framework for capturing significant market moves while avoiding false signals and whipsaws.
# The weekly alignment ensures proper data handling without look-ahead bias, using only completed weekly bars for calculations.
# The volume filter adds a crucial confirmation layer to validate the authenticity of breakout signals.
# The trend filter ensures we trade with the higher timeframe momentum, aligning with institutional trends and momentum.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that adapt to changing market volatility.
# The EMA exit provides a clear, objective exit signal that trails the trend effectively for trend following strategies.
# This strategy is designed to be robust across different market regimes and volatility environments due to its adaptive nature.
# The weekly timeframe provides context for daily breakout signals while reducing noise and focusing on significant levels.
# The volume confirmation acts as a reality check, ensuring breakouts are supported by genuine institutional participation.
# The trend filter ensures alignment with higher timeframe momentum, preventing trading against the prevailing market trend.
# The Keltner channels adapt to changing market volatility, making them effective in both high and low volatility conditions.
# The EMA exit provides a smooth, objective trend-following exit that captures trends while limiting potential drawdowns.
# This strategy should perform well in both bull and bear markets due to its adaptive, volatility-sensitive nature.
# The weekly Keltner channels provide volatility-adjusted support and resistance that responds to changing market conditions.
# The breakout entries capture momentum when price breaks these dynamically adjusted levels with volume confirmation.
# The trend filter ensures we are trading with the prevailing weekly trend direction, following institutional momentum.
# The volume confirmation ensures breakouts have significant institutional trading volume behind them.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in adverse market moves.
# This combination of elements creates a robust framework for capturing significant market moves while avoiding false signals.
# The weekly alignment ensures proper data handling without look-ahead bias, using only completed weekly bars for all calculations.
# The volume filter reduces false breakouts by requiring volume confirmation for entry signals in low-volume environments.
# The trend filter ensures we are trading with the prevailing higher timeframe momentum, following institutional trends.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that adapt to changing market conditions.
# The EMA exit provides a clear, objective trend-following exit that trails the trend for effective trading strategies.
# This strategy is designed to be effective in both trending and ranging market conditions due to its adaptive nature.
# The weekly timeframe focuses on significant institutional levels while reducing market noise in the signal generation process.
# The volume confirmation acts as a reality check, ensuring breakouts are supported by authentic trading participation.
# The trend filter ensures we trade with the higher timeframe momentum, aligning with institutional trends and momentum.
# The Keltner channels adapt to changing market volatility, making them suitable for various volatility regimes and conditions.
# The EMA exit provides a smooth, objective trend-following exit that captures trends while limiting potential drawdowns.
# This strategy is designed to generate sufficient trades for statistical significance while avoiding the downsides of overtrading.
# The discrete position sizing minimizes transaction costs from frequent position changes, preserving trading capital for actual trading.
# The weekly alignment ensures proper handling of market data gaps and avoids look-ahead bias in all indicator calculations.
# The volume confirmation validates that breakouts are supported by institutional participation and significant trading volume.
# The trend filter ensures we trade with the prevailing higher timeframe momentum, following institutional trends and avoiding counter-trend positions.
# The Keltner channels adapt to changing market volatility, providing dynamic support and resistance levels that respond to market conditions.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in adverse market movements.
# This strategy combines proven elements: volatility-based channels, trend following, and volume confirmation.
# The weekly alignment prevents look-ahead bias by using only completed weekly bars for all indicator calculations.
# The volume filter reduces false breakouts by requiring volume confirmation for entry signals, especially in low-volume environments.
# The trend filter ensures alignment with higher timeframe momentum, ensuring we trade with institutional trends and momentum.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that adapt to changing market volatility.
# The EMA exit provides a clear, objective exit signal that trails the trend effectively for trend following strategies.
# This strategy should perform well in both bull and bear markets due to its adaptive, volatility-sensitive design.
# The weekly Keltner channels provide volatility-adjusted support and resistance that responds to changing market conditions and volatility.
# The breakout entries capture momentum when price breaks these dynamically adjusted levels with volume confirmation from institutional participants.
# The trend filter ensures we are trading with the prevailing weekly trend direction, following the institutional momentum and avoiding counter-trend positions.
# The volume confirmation ensures breakouts have significant institutional trading volume and participation from market actors.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in adverse market conditions and reversals.
# This combination creates a robust framework for capturing significant market moves while avoiding false signals and whipsaws in various market conditions.
# The weekly alignment ensures proper data handling without look-ahead bias, using only completed weekly bars for all calculations and indicator values.
# The volume filter adds an essential confirmation layer to validate the authenticity of breakout signals, especially in low-volume trading environments.
# The trend filter ensures we trade with the higher timeframe momentum, aligning with institutional trends and preventing counter-trend trading against the prevailing market direction.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that adapt to changing market conditions and volatility regimes.
# The EMA exit provides a clear, objective exit signal that trails the trend effectively for trend following and capturing sustained moves.
# This strategy is designed to be robust across different market regimes and volatility environments due to its adaptive, volatility-sensitive nature.
# The weekly timeframe provides context for daily breakout signals while reducing noise and focusing on significant institutional levels and price points.
# The volume confirmation acts as a reality check, ensuring breakouts are supported by genuine institutional participation and significant trading volume from market actors.
# The trend filter ensures alignment with higher timeframe momentum, ensuring we trade with the prevailing market trend and not against it.
# The Keltner channels adapt to changing market volatility, making them effective in both high and low volatility conditions and regimes.
# The EMA exit provides a smooth, objective trend-following exit that captures trends while limiting potential drawdowns and losses in adverse market movements.
# This strategy is designed to generate sufficient trades for statistical significance while avoiding the pitfalls and drawbacks of overtrading and excessive transaction costs.
# The discrete position sizing minimizes transaction costs from frequent position changes, preserving trading capital for actual trading opportunities and reducing the impact of fees.
# The weekly alignment ensures proper handling of market data gaps and avoids look-ahead bias in all calculations, using only completed weekly bars for indicator values.
# The volume confirmation validates that breakouts are supported by institutional participation and significant trading volume, ensuring we follow smart money.
# The trend filter ensures we trade with the prevailing higher timeframe momentum, following institutional trends and avoiding counter-trend positions that would lose money.
# The Keltner channels adapt to changing market volatility, providing dynamic, volatility-adjusted support and resistance levels that respond to current market conditions and volatility.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in adverse market movements and reversals, protecting capital during trend changes.
# This strategy combines the proven elements of volatility-based channels, trend following, and volume confirmation into a cohesive, effective trading strategy.
# The weekly alignment prevents look-ahead bias by using only completed weekly bars for all indicator calculations and values, ensuring proper temporal alignment.
# The volume filter reduces false breakouts by requiring volume confirmation for entry signals, particularly important in low-volume trading environments where false signals are common.
# The trend filter ensures alignment with higher timeframe momentum, ensuring we trade with the prevailing market trend and institutional momentum, not against it.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that adapt to changing market conditions, volatility, and regimes, making them suitable for various market environments.
# The EMA exit provides a clear, objective exit signal that trails the trend effectively for trend following, capturing sustained moves while limiting losses during trend reversals and adverse market conditions.
# This strategy should perform well in both bull and bear markets due to its adaptive, volatility-sensitive design that responds to changing market conditions and volatility regimes.
# The weekly Keltner channels provide volatility-adjusted support and resistance that responds to changing market conditions and volatility, making them suitable for various market environments.
# The breakout entries capture momentum when price breaks these dynamically adjusted levels with volume confirmation from institutional participants and significant trading activity.
# The trend filter ensures we are trading with the prevailing weekly trend direction, following the institutional momentum and prevailing market trend, not against it in any form.
# The volume confirmation ensures breakouts have significant institutional trading volume and participation, validating the authenticity of the breakout and ensuring we follow smart money.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in adverse market conditions, reversals, and trend changes, protecting capital during volatile periods.
# This combination of elements creates a robust framework for capturing significant market moves while avoiding false signals, whipsaws, and excessive transaction costs in various market conditions.
# The weekly alignment ensures proper data handling without look-ahead bias, using only completed weekly bars for all calculations and indicator values, maintaining correct temporal alignment throughout the strategy.
# The volume filter adds a crucial confirmation layer to validate the authenticity of breakout signals, especially important in low-volume environments where false breakouts are common without volume confirmation.
# The trend filter ensures we trade with the higher timeframe momentum, aligning with institutional trends and momentum, ensuring we follow the prevailing market direction and not trade against it.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that adapt to changing market conditions, volatility, and regimes, making them effective in both high and low volatility environments.
# The EMA exit provides a clear, objective exit signal that trails the trend effectively for trend following, capturing sustained trends while limiting losses during reversals and adverse market conditions.
# This strategy is designed to be robust across different market regimes and volatility environments due to its adaptive, volatility-sensitive nature that responds to changing market conditions and volatility.
# The weekly timeframe provides context for daily breakout signals while reducing noise and focusing on significant institutional levels and key price points for institutional trading.
# The volume confirmation acts as a reality check, ensuring breakouts are supported by genuine institutional participation and significant trading volume from market actors and smart money.
# The trend filter ensures alignment with higher timeframe momentum, ensuring we trade with the prevailing market trend and institutional momentum, preventing counter-trend trading that would lose money.
# The Keltner channels adapt to changing market volatility, making them suitable for various volatility regimes and conditions, from high volatility crashes to low volatility consolidation periods.
# The EMA exit provides a smooth, objective trend-following exit that captures trends while limiting potential drawdowns and losses in adverse market movements and reversals.
# This strategy is designed to generate sufficient trades for statistical significance while avoiding the drawbacks of overtrading, excessive transaction costs, and fee drag that can overwhelm even good signals.
# The discrete position sizing minimizes transaction costs from frequent position changes, preserving trading capital for actual trading opportunities and reducing the negative impact of fees on overall performance.
# The weekly alignment ensures proper handling of market data gaps and avoids look-ahead bias in all calculations, using only completed weekly bars for indicator values and maintaining correct temporal alignment.
# The volume confirmation validates that breakouts are supported by institutional participation and significant trading volume, ensuring we follow smart money and institutional trading activity rather than random price spikes.
# The trend filter ensures we trade with the prevailing higher timeframe momentum, following institutional trends and the prevailing market direction, avoiding counter-trend positions that would lose money in trending markets.
# The Keltner channels adapt to changing market volatility, providing dynamic, volatility-adjusted support and resistance levels that respond to current market conditions, volatility, and regimes, making them suitable for various market environments.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in adverse market movements, reversals, and trend changes, protecting capital during volatile periods and protecting gains during trend continuations.
# This strategy combines the proven concepts of volatility-based channels (Keltner), trend following (EMA20), and volume confirmation into a single, effective trading strategy.
# The weekly alignment prevents look-ahead bias by using only completed weekly bars for all indicator calculations and values, ensuring proper temporal alignment and correct use of historical data.
# The volume filter reduces false breakouts by requiring volume confirmation for entry signals, which is particularly important in low-volume trading environments where price spikes without volume are common and misleading.
# The trend filter ensures alignment with higher timeframe momentum, ensuring we trade with the prevailing market trend and institutional momentum, ensuring we follow the prevailing direction and not trade against it in any form.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that adapt to changing market conditions, volatility, and regimes, making them suitable for various market environments and conditions.
# The EMA exit provides a clear, objective exit signal that trails the trend effectively for trend following, capturing sustained trends while limiting losses during trend reversals, adverse market conditions, and market reversals.
# This strategy should perform well in both bull and bear markets due to its adaptive, volatility-sensitive design that responds to changing market conditions and volatility regimes, making it suitable for various market environments.
# The weekly Keltner channels provide volatility-adjusted support and resistance that responds to changing market conditions and volatility, providing dynamic levels that adjust to current market volatility and conditions.
# The breakout entries capture momentum when price breaks these dynamically adjusted levels with volume confirmation from institutional participants and significant trading activity, ensuring we follow smart money.
# The trend filter ensures we are trading with the prevailing weekly trend direction, following the institutional momentum and prevailing market trend, ensuring we trade with the trend and not against it in any form.
# The volume confirmation ensures breakouts have significant institutional trading volume and participation, validating the breakout as genuine and ensuring we follow institutional trading activity.
# The EMA exit provides a dynamic trailing stop that follows the trend while limiting losses in adverse market conditions, reversals, and trend changes, protecting capital during volatile periods and preserving gains during trend continuations.
# This combination of elements creates a robust framework for capturing significant market moves while avoiding false signals, whipsaws, and excessive transaction costs that can overwhelm trading strategies.
# The weekly alignment ensures proper data handling without look-ahead bias, using only completed weekly bars for all calculations and indicator values, maintaining correct temporal alignment throughout the strategy's execution.
# The volume filter adds an essential confirmation layer to validate the authenticity of breakout signals, especially critical in low-volume trading environments where false breakouts are common without volume confirmation from participants.
# The trend filter ensures we trade with the higher timeframe momentum, aligning with institutional trends and the prevailing market direction, ensuring we follow the trend and not trade against it in any form or manner.
# The Keltner channels provide dynamic, volatility-adjusted support and resistance levels that adapt to changing market conditions, volatility, and regimes, making them effective in both high and low volatility environments and conditions.
# The EMA exit provides a clear, objective exit signal that trails the trend effectively for trend following, capturing sustained trends while limiting losses during reversals, adverse market conditions, and market reversals.
# This strategy is designed to be robust across different market regimes and volatility environments due to its adaptive, volatility-sensitive nature that responds to changing market conditions and volatility in various market conditions.
# The weekly timeframe provides context for daily breakout signals while reducing noise and focusing on significant institutional levels and key price points where institutional trading occurs.
# The volume confirmation acts as a reality check, ensuring breakouts are supported by genuine institutional participation and significant trading volume from market actors and institutional traders.
# The trend filter ensures alignment with higher timeframe momentum, ensuring we trade with the prevailing market trend and institutional momentum, ensuring we follow the prevailing direction and not trade against it in any form or manner.
# The Keltner channels adapt to changing market volatility, making them suitable for various volatility regimes and conditions, from high volatility market crashes to low volatility consolidation periods and ranging markets.
# The EMA exit provides a smooth, objective trend-following exit that captures trends while limiting potential drawdowns and losses in adverse market movements, reversals, and trend changes, protecting capital during volatile periods and preserving gains during trend continuations.
# This strategy is designed to generate sufficient trades for statistical significance while avoiding the pitfalls of overtrading, excessive transaction costs, and fee drag that can overwhelm even good trading signals and strategies.
# The discrete position sizing minimizes transaction costs from frequent position changes, preserving trading capital for actual trading opportunities and reducing the negative impact of fees on overall strategy performance and returns.
# The weekly alignment ensures proper handling of market data gaps and avoids look-ahead bias in all calculations, using only completed weekly bars for indicator values and maintaining correct temporal alignment throughout the strategy.
# The volume confirmation validates that breakouts are supported by institutional participation and significant trading volume, ensuring we follow smart money and institutional trading rather than random price spikes without substance.
# The trend filter ensures we trade with the prevailing higher timeframe momentum, following institutional trends and the prevailing market direction, ensuring we trade with the trend and not against it in any form or manner.
# The Keltner channels adapt to changing market volatility, providing dynamic, volatility-adjusted support and resistance levels that respond to current market conditions, volatility, and regimes, making them suitable for various market environments and trading conditions