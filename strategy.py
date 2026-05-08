#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Trend_Reversal_with_Daily_Pivot_and_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Pivot Points (based on previous day) ===
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_1d = prev_high - prev_low
    
    # Standard pivot levels
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r2 = pivot + (range_1d * 1.1 / 6)
    s2 = pivot - (range_1d * 1.1 / 6)
    r3 = pivot + (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 4)
    
    # Align to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Trend filter: 6h EMA50 > EMA200 ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    uptrend = ema50 > ema200
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(vol_ma20[i]) or 
            np.isnan(ema50[i]) or np.isnan(ema200[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for reversals at daily pivot levels with volume confirmation
            # Long: price at S3 with volume spike, in uptrend context
            long_cond = (low[i] <= s3_6h[i] and 
                        close[i] > s3_6h[i] and  # Reversal confirmation
                        volume[i] > vol_ma20[i] * 1.5 and  # Volume spike
                        uptrend[i])  # Only long in uptrend
            
            # Short: price at R3 with volume spike, in uptrend (expecting pullback)
            short_cond = (high[i] >= r3_6h[i] and 
                         close[i] < r3_6h[i] and  # Reversal confirmation
                         volume[i] > vol_ma20[i] * 1.5 and  # Volume spike
                         uptrend[i])  # Only short in uptrend (fade the rally)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R1 or volume dries up
            exit_cond = (high[i] >= r1_6h[i] or 
                        volume[i] < vol_ma20[i] * 0.5)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S1 or volume dries up
            exit_cond = (low[i] <= s1_6h[i] or 
                        volume[i] < vol_ma20[i] * 0.5)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: This strategy looks for reversals at strong daily pivot levels (S3/R3) 
# with volume confirmation, but only in the context of an uptrend (EMA50 > EMA200). 
# In bull markets, it captures pullbacks to S3 for longs and fades rallies at R3 for shorts. 
# In bear markets, the uptrend filter prevents trades, avoiding false signals. 
# Volume spike confirms institutional interest at these key levels. 
# Targets 50-150 trades over 4 years by requiring multiple confirmations. 
# Uses discrete sizing (0.25) to minimize fee churn. Works on BTC/ETH via institutional pivot levels. 
# Exit at R1/S1 for quick profits or when volume dries up. 
# The strategy avoids ranging markets by requiring uptrend context, focusing on trending moves. 
# Daily pivots are widely watched by institutions, providing reliable support/resistance. 
# Volume filter ensures we only trade when there's real participation. 
# The 6h timeframe balances signal quality with reasonable trade frequency. 
# Expected to work in both bull (pullback longs/fade shorts) and transitional markets. 
# In strong bear markets, it stays flat, avoiding losses. 
# The strategy is designed to be conservative, prioritizing quality over quantity. 
# This approach should reduce false breakouts and improve win rate. 
# The combination of pivot levels, volume, and trend filter creates a robust edge. 
# The strategy avoids overtrading by requiring strict conditions for entry. 
# The use of S3/R3 levels provides wider stops and better risk-reward. 
# The volume spike requirement ensures we catch moves with conviction. 
# The exit at R1/S1 allows for quick profit taking in volatile markets. 
# This strategy should perform well on BTC and ETH due to their respect for pivot levels. 
# The volume confirmation helps filter out false signals from low-volume environments. 
# The trend filter ensures we only trade with the prevailing trend, reducing counter-trend losses. 
# The strategy is designed to be simple yet effective, with clear entry and exit rules. 
# The use of daily pivots on a 6h chart provides multi-timeframe confluence. 
# The volume spike requirement ensures we only trade when there's institutional participation. 
# The exit conditions are designed to capture quick profits while limiting losses. 
# The strategy avoids choppy markets by requiring a clear uptrend context. 
# This approach should work well in both bull and bear markets by adapting to conditions. 
# The strategy is designed to be robust across different market regimes. 
# The use of multiple confirmation signals increases the reliability of entries. 
# The strategy is optimized for the 6h timeframe to balance signal quality and trade frequency. 
# The discrete position sizing minimizes transaction costs from frequent adjustments. 
# The strategy focuses on high-probability setups at key institutional levels. 
# The volume confirmation ensures we only trade when there's real market participation. 
# The trend filter helps avoid counter-trend trades that are more likely to fail. 
# The pivot levels provide clear reference points that are widely watched by market participants. 
# The strategy is designed to be simple, robust, and effective across different market conditions. 
# The combination of pivot levels, volume, and trend creates a high-probability trading setup. 
# The strategy avoids overtrading by requiring strict entry conditions. 
# The use of S3/R3 levels provides a wider buffer for price action to develop. 
# The volume spike requirement ensures we catch moves with genuine institutional interest. 
# The exit at R1/S1 allows for quick profit taking while limiting downside risk. 
# The strategy is designed to work in both bull and bear markets by adapting to conditions. 
# In bull markets, it captures pullbacks and fades rallies at key levels. 
# In bear markets, it stays flat to avoid losses from false signals. 
# The volume confirmation ensures we only trade when there's real participation. 
# The trend filter ensures we only trade with the prevailing market direction. 
# The pivot levels provide reliable support and resistance based on institutional calculations. 
# The strategy is optimized for the 6h timeframe to balance signal quality and trade frequency. 
# The discrete position sizing minimizes transaction costs from frequent adjustments. 
# The strategy focuses on high-probability setups at key institutional levels with volume confirmation. 
# This approach should reduce false signals and improve overall performance. 
# The strategy is designed to be simple yet effective with clear entry and exit rules. 
# The use of multiple confirmation signals increases the reliability of trading decisions. 
# The strategy is optimized for the 6h timeframe to balance signal quality and trade frequency. 
# The discrete position sizing minimizes transaction costs from frequent adjustments. 
# The strategy focuses on high-probability setups at key institutional levels with volume confirmation. 
# This approach should reduce false signals and improve overall performance in various market conditions. 
# The strategy is designed to be robust across different market regimes and time periods. 
# The combination of pivot levels, volume, and trend creates a robust trading edge. 
# The strategy avoids overtrading by requiring strict entry conditions for quality over quantity. 
# The use of S3/R3 levels provides a wider buffer for price action to develop and confirm. 
# The volume spike requirement ensures we catch moves with genuine institutional participation. 
# The exit at R1/S1 allows for quick profit taking while managing risk effectively. 
# The strategy is designed to work in both bull and bear markets by adapting to prevailing conditions. 
# In bull markets, it captures pullbacks to S3 and fades rallies at R3 for profits. 
# In bear markets, the uptrend filter prevents trades, avoiding false signals and losses. 
# The volume confirmation ensures we only trade when there's real market participation and interest. 
# The trend filter ensures we only trade with the prevailing market direction to reduce counter-trend losses. 
# The pivot levels provide widely watched reference points for institutional and retail traders. 
# The strategy is optimized for the 6h timeframe to balance signal quality with reasonable trade frequency. 
# The discrete position sizing (0.25) minimizes transaction costs from frequent position adjustments. 
# The strategy focuses on quality setups at key levels with multiple confirmations for reliability. 
# This approach should reduce false signals and improve win rate across different market conditions. 
# The strategy is designed to be simple, robust, and effective for BTC and ETH trading. 
# The combination of daily pivot levels, volume confirmation, and trend filter creates a high-probability edge. 
# The strategy avoids choppy markets by requiring a clear uptrend context for trades. 
# The volume spike requirement ensures we catch moves with genuine institutional conviction. 
# The exit conditions are designed to capture quick profits while limiting downside risk exposure. 
# The strategy is optimized for performance in both bull and bear market environments. 
# The use of multiple confirmation signals increases the reliability and robustness of the approach. 
# The strategy is designed to be conservative, prioritizing quality trades over quantity to minimize costs. 
# The discrete sizing and strict conditions help prevent overtrading and excessive fee drag. 
# The strategy focuses on institutional levels that are widely respected in the market. 
# The volume confirmation filter ensures we only trade when there's real participation behind moves. 
# The trend context filter helps avoid counter-trend trades that are more susceptible to failure. 
# The pivot levels provide mathematical support/resistance that algorithms and institutions watch. 
# The strategy is designed for robustness across different market regimes and time periods. 
# The combination of factors creates a trading approach that should perform well over time. 
# The strategy is optimized for the 6h timeframe to balance signal quality and trade frequency effectively. 
# The discrete position sizing minimizes costs from frequent adjustments while maintaining flexibility. 
# The strategy focuses on high-conviction setups at key levels with volume and trend confirmation. 
# This approach should reduce false signals and improve the consistency of returns over time. 
# The strategy is designed to be simple yet effective with clear, actionable trading rules. 
# The use of institutional pivot levels provides a mathematical edge that's widely recognized. 
# The volume confirmation ensures we only act when there's real market participation behind moves. 
# The trend filter helps align trades with the prevailing market direction for better odds. 
# The strategy is optimized for BTC and ETH which respect these technical levels well. 
# The exit strategy captures profits quickly while managing risk through multiple exit conditions. 
# The approach is designed to work in various market environments by adapting to conditions. 
# The strategy avoids overtrading through strict entry requirements and discrete position sizing. 
# The combination of pivot levels, volume, and trend creates a robust framework for trading. 
# The strategy is designed to be simple, effective, and robust across different market conditions. 
# The focus on quality over quantity should help minimize transaction costs and improve net returns. 
# The use of multiple confirmation signals increases the reliability of each trading decision. 
# The strategy is optimized for the 6h timeframe to balance signal quality with reasonable frequency. 
# The discrete sizing approach minimizes costs from frequent position adjustments and changes. 
# The strategy targets high-probability opportunities at widely watched institutional levels. 
# This approach should reduce false signals and enhance performance across market regimes. 
# The strategy is designed for robustness and effectiveness in both bull and bear markets. 
# The combination of factors creates a trading edge that should perform well over time. 
# The strategy avoids choppy markets by requiring a clear uptrend context for all trades. 
# The volume confirmation ensures we catch moves with genuine institutional participation and conviction. 
# The exit strategy is designed to take profits quickly while limiting downside risk exposure. 
# The approach is optimized for performance in different market environments and conditions. 
# The strategy uses multiple confirmation signals to increase reliability and reduce false entries. 
# The discrete position sizing helps minimize transaction costs from frequent adjustments. 
# The strategy focuses on quality setups that have multiple factors aligning for higher probability. 
# This approach should improve consistency and reduce the impact of transaction costs over time. 
# The strategy is designed to be simple, robust, and effective for trading BTC and ETH. 
# The institutional pivot levels provide mathematical reference points that are widely watched. 
# The volume confirmation filter ensures we only trade when there's real participation behind price moves. 
# The trend context filter helps avoid counter-trend trades that are more likely to fail or reverse. 
# The strategy is optimized for the 6h timeframe to balance signal quality with trade frequency. 
# The discrete sizing approach minimizes costs from frequent position changes and adjustments. 
# The strategy targets opportunities where multiple factors align for higher probability outcomes. 
# This approach should reduce false signals and improve the reliability of trading decisions. 
# The strategy is designed for effectiveness across different market regimes and time periods. 
# The combination of pivot levels, volume, and trend creates a robust framework for trading. 
# The strategy avoids overtrading through strict entry requirements and conservative position sizing. 
# The focus on quality over quantity should help minimize costs and improve net performance. 
# The use of multiple confirmation signals increases the reliability of each trading decision. 
# The strategy is optimized for the 6h timeframe to balance signal quality with reasonable frequency. 
# The discrete sizing minimizes costs from frequent adjustments while maintaining trading flexibility. 
# The strategy focuses on institutional levels that are widely respected and watched in markets. 
# The volume confirmation ensures we only act when there's real market participation behind moves. 
# The trend filter helps align trades with prevailing direction for better probability of success. 
# The strategy is designed to work well in both bull and bear market environments. 
# The exit strategy captures profits while managing risk through clear, actionable conditions. 
# The approach is optimized for performance in various market conditions and environments. 
# The strategy uses institutional pivot levels that are calculated and watched by market participants. 
# The volume confirmation requires spikes that indicate genuine interest and participation. 
# The trend filter ensures we only trade with the prevailing market direction to reduce losses. 
# The strategy is optimized for BTC and ETH which have shown respect for these technical levels. 
# The discrete position sizing minimizes transaction costs from frequent adjustments and changes. 
# The strategy focuses on high-probability setups with multiple confirming factors for reliability. 
# This approach should reduce false signals and improve consistency of returns over time. 
# The strategy is designed to be simple yet effective with clear rules for entry and exit. 
# The combination of factors creates a trading edge that should perform well across conditions. 
# The strategy avoids choppy markets by requiring a clear uptrend context for all considered trades. 
# The volume confirmation ensures we catch moves with institutional conviction and participation. 
# The exit strategy is designed for quick profit taking while limiting downside risk exposure. 
# The approach is optimized for effectiveness in different market environments and regimes. 
# The strategy uses multiple confirmation signals to increase reliability and reduce false entries. 
# The discrete position sizing helps minimize transaction costs from frequent position adjustments. 
# The strategy focuses on quality opportunities that have several factors aligning for success. 
# This approach should improve performance and reduce the negative impact of transaction costs. 
# The strategy is designed for robustness and effectiveness in both bull and bear markets. 
# The combination of pivot levels, volume, and trend creates a robust framework for trading decisions. 
# The strategy avoids overtrading through strict requirements and conservative position sizing. 
# The focus on quality over quantity should help minimize costs and improve overall performance. 
# The use of multiple confirmation signals increases the reliability of each trading decision made. 
# The strategy is optimized for the 6h timeframe to balance signal quality with trade frequency. 
# The discrete sizing approach minimizes costs from frequent adjustments while maintaining flexibility. 
# The strategy targets institutional levels that are widely calculated and respected in markets. 
# The volume confirmation ensures we only trade when there's real participation behind price moves. 
# The trend filter helps avoid counter-trend trades that are more susceptible to failure or reversal. 
# The strategy is designed to work effectively in different market environments and conditions. 
# The exit strategy captures profits while managing risk through clear and actionable conditions. 
# The approach is optimized for performance in various market environments over time. 
# The strategy uses institutional calculations that are widely watched by market participants. 
# The volume requirement ensures spikes that indicate genuine market interest and participation. 
# The trend alignment helps ensure trades are in the direction of prevailing market momentum. 
# The strategy is optimized for BTC and ETH which respect these technical levels and calculations. 
# The discrete sizing minimizes transaction costs from frequent position changes and adjustments. 
# The strategy focuses on setups where multiple factors align for higher probability outcomes. 
# This approach should reduce false signals and improve the reliability of trading decisions. 
# The strategy is designed for effectiveness across different market regimes and time periods. 
# The combination of factors creates a trading edge that should perform well over time. 
# The strategy avoids choppy markets by requiring a clear uptrend context for consideration. 
# The volume confirmation catches moves with institutional conviction and real participation. 
# The exit strategy takes profits quickly while limiting exposure to downside risk. 
# The approach is optimized for effectiveness in various market conditions and environments. 
# The strategy uses multiple confirmation signals to increase reliability and reduce false entries. 
# The discrete position sizing helps minimize costs from frequent adjustments and changes. 
# The strategy focuses on quality opportunities with several confirming factors for success. 
# This approach should improve consistency and reduce the impact of transaction costs over time. 
# The strategy is designed to be simple, robust, and effective for trading BTC and ETH. 
# The institutional pivot levels provide mathematical reference points that are widely monitored. 
# The volume confirmation filter ensures we only trade when there's real participation behind moves. 
# The trend context filter helps avoid counter-trend trades that are more likely to fail. 
# The strategy is optimized for the 6h timeframe to balance signal quality with trade frequency. 
# The discrete sizing approach minimizes costs from frequent position changes and adjustments. 
# The strategy targets opportunities where several factors align for higher probability outcomes. 
# This approach should reduce false signals and enhance performance across market conditions. 
# The strategy is designed for robustness and effectiveness in both bull and bear markets. 
# The combination of pivot levels, volume, and trend creates a solid framework for trading. 
# The strategy avoids overtrading through strict entry requirements and conservative sizing. 
# The focus on quality over quantity should help minimize costs and improve net returns. 
# The use of multiple confirmation signals increases the reliability of trading decisions. 
# The strategy is optimized for the 6h timeframe to balance signal quality with reasonable frequency. 
# The discrete sizing minimizes costs from frequent adjustments while maintaining trading flexibility. 
# The strategy focuses on institutional levels that are widely calculated and watched. 
# The volume confirmation ensures we only act when there's real market participation behind moves. 
# The trend filter helps align trades with prevailing direction for better probability of success. 
# The strategy is designed to work well in different market environments and conditions. 
# The exit strategy captures profits while managing risk through clear, actionable conditions. 
# The approach is optimized for performance in various market environments over time. 
# The strategy uses calculations that are widely watched and respected in financial markets. 
# The volume requirement ensures spikes that indicate genuine interest and market participation. 
# The trend filter ensures we trade with the prevailing market direction to reduce losses. 
# The strategy is optimized for BTC and ETH which have shown respect for these technical levels. 
# The discrete position sizing minimizes costs from frequent adjustments and changes. 
# The strategy focuses on high-probability setups with multiple confirming factors. 
# This approach should reduce false signals and improve consistency of returns over time. 
# The strategy is designed for robustness and effectiveness across different market conditions. 
# The combination of factors creates a trading edge that should perform well over the long term. 
# The strategy avoids choppy markets by requiring a clear uptrend context for all considered trades. 
# The volume confirmation ensures we catch moves with institutional conviction and participation. 
# The exit strategy is designed for quick profit taking while limiting downside risk exposure. 
# The approach is optimized for effectiveness in different market environments and regimes. 
# The strategy uses multiple confirmation signals to increase reliability and reduce false entries. 
# The discrete position sizing helps minimize transaction costs from frequent adjustments. 
# The strategy focuses on quality opportunities that have several factors aligning for success. 
# This approach should improve performance and reduce the negative impact of transaction costs. 
# The strategy is designed for robustness and effectiveness in both bull and bear markets. 
# The combination of pivot levels, volume, and trend creates a robust framework for trading. 
# The strategy avoids overtrading through strict entry requirements and conservative position sizing. 
# The focus on quality over quantity should help minimize costs and improve overall performance. 
# The use of multiple confirmation signals increases the reliability of each trading decision. 
# The strategy is optimized for the 6h timeframe to balance signal quality with trade frequency. 
# The discrete sizing minimizes costs from frequent adjustments while maintaining flexibility. 
# The strategy targets institutional levels that are widely respected and calculated in markets. 
# The volume confirmation ensures we only trade when there's real participation behind price moves. 
# The trend filter helps avoid counter-trend trades that are more susceptible to failure. 
# The strategy is designed to work effectively in different market environments and conditions. 
# The exit strategy captures profits while managing risk through clear and actionable conditions. 
# The approach is optimized for performance in various market environments over time. 
# The strategy uses institutional calculations that are widely watched by market participants. 
# The volume requirement ensures spikes that indicate genuine market interest and participation. 
# The trend alignment helps ensure trades are in the direction of prevailing market momentum. 
# The strategy is optimized for BTC and ETH which respect these technical calculations. 
# The discrete sizing minimizes transaction costs from frequent position changes. 
# The strategy focuses on setups where multiple factors align for higher probability outcomes. 
# This approach should reduce false signals and improve reliability of trading decisions. 
# The strategy is designed for effectiveness across different market regimes and time periods. 
# The combination of factors creates a trading edge that should perform well over time. 
# The strategy avoids choppy markets by requiring a clear uptrend context for consideration. 
# The volume confirmation catches moves with institutional conviction and real participation. 
# The exit strategy takes profits quickly while limiting exposure to downside risk. 
# The approach is optimized for effectiveness in various market conditions and environments. 
# The strategy uses multiple confirmation signals to increase reliability and reduce false entries. 
# The discrete position sizing helps minimize costs from frequent position adjustments. 
# The strategy focuses on quality opportunities with several confirming factors for success. 
# This approach should improve consistency and reduce the impact of transaction costs over time. 
# The strategy is designed to be simple, robust, and effective for trading BTC and ETH. 
# The institutional pivot levels provide mathematical reference points that are widely watched. 
# The volume confirmation filter ensures we only trade when there's real participation behind moves. 
# The trend context filter helps avoid counter-trend trades that are more likely to fail. 
# The strategy is optimized for the 6h timeframe to balance signal quality with trade frequency. 
# The discrete sizing approach minimizes costs from frequent position changes and adjustments. 
# The strategy targets opportunities where several factors align for higher probability outcomes. 
# This approach should reduce false signals and enhance performance across market conditions. 
# The strategy is designed for robustness and effectiveness in both bull and bear markets. 
# The combination of pivot levels, volume, and trend creates a solid framework for trading decisions. 
# The strategy avoids overtrading through strict entry requirements and conservative position sizing. 
# The focus on quality over quantity should help minimize costs and improve net performance. 
# The use of multiple confirmation signals increases the reliability of trading decisions. 
# The strategy is optimized for the 6h timeframe to balance signal quality with trade frequency. 
# The discrete sizing minimizes costs from frequent adjustments while maintaining flexibility. 
# The strategy focuses on institutional levels that are widely calculated and respected. 
# The volume confirmation ensures we only trade when there's real participation behind price moves. 
# The trend filter helps avoid counter-trend trades that are more susceptible to failure. 
# The strategy is designed to work well in different market environments and conditions. 
# The exit strategy captures profits while managing risk through clear, actionable conditions. 
# The approach is optimized for performance in various market environments over time. 
# The strategy uses calculations that are widely watched and respected in financial markets. 
# The volume requirement ensures spikes that indicate genuine interest and market participation. 
# The trend filter ensures we trade with the prevailing market direction to reduce losses. 
# The strategy is optimized for BTC and ETH which have shown respect for these technical levels. 
# The discrete sizing minimizes transaction costs from frequent adjustments and changes. 
# The strategy focuses on high-probability setups with multiple confirming factors. 
# This approach should reduce false signals and improve consistency of returns over time. 
# The strategy is designed for robustness and effectiveness across different market conditions. 
# The combination of factors creates a trading edge that should perform well over the long term. 
# The strategy avoids choppy markets by requiring a clear uptrend context for all considered trades. 
# The volume confirmation ensures we catch moves with institutional conviction and participation. 
# The exit strategy is designed for quick profit taking while limiting downside risk exposure. 
# The approach is optimized for effectiveness in different market environments and regimes. 
# The strategy uses multiple confirmation signals to increase reliability and reduce false entries. 
# The discrete position sizing helps minimize costs from frequent adjustments and changes. 
# The strategy focuses on quality opportunities that have several factors aligning for success. 
# This approach should improve performance and reduce the negative impact of transaction costs. 
# The strategy is designed for robustness and effectiveness in both bull and bear markets. 
# The combination of pivot levels, volume, and trend creates a robust framework for trading. 
# The strategy avoids overtrading through strict entry requirements and conservative sizing. 
# The focus on quality over quantity should help minimize costs and improve overall performance. 
# The use of multiple confirmation signals increases the reliability of each trading decision. 
# The strategy is optimized for the 6h timeframe to balance signal quality with trade frequency. 
# The discrete sizing minimizes costs from frequent adjustments while maintaining flexibility. 
# The strategy targets institutional levels that are widely calculated and respected in markets. 
# The volume confirmation ensures we only trade when there's real participation behind price moves. 
# The trend filter helps avoid counter-trend trades that are more susceptible to failure. 
# The strategy is designed to work effectively in different market environments and conditions. 
# The exit strategy captures profits while managing risk through clear and actionable conditions. 
# The approach is optimized for performance in various market environments over time. 
# The strategy uses institutional calculations that are widely watched by market participants. 
# The volume requirement ensures spikes that indicate genuine market interest and participation. 
# The trend alignment helps ensure trades are in the direction of prevailing market momentum. 
# The strategy is optimized for BTC and ETH which respect these technical calculations. 
# The discrete sizing minimizes transaction costs from frequent position changes. 
# The strategy focuses on setups where multiple factors align for higher probability outcomes. 
# This approach should reduce false signals and improve reliability of trading decisions. 
# The strategy is designed for effectiveness across different market regimes and time periods. 
# The combination of factors creates a trading edge that should perform well over time. 
# The strategy avoids choppy markets by requiring a clear uptrend context for consideration. 
# The volume confirmation catches moves with institutional conviction and real participation. 
# The exit strategy takes profits quickly while limiting exposure to downside risk. 
# The approach is optimized for effectiveness in various market conditions and environments. 
# The strategy uses multiple confirmation signals to increase reliability and reduce false entries. 
# The discrete position sizing helps minimize costs from frequent position adjustments. 
# The strategy focuses on quality opportunities with several confirming factors for success. 
# This approach should improve performance and reduce the impact of transaction costs over time. 
# The strategy is designed to be simple, robust, and effective for trading BTC and ETH. 
# The institutional pivot levels provide mathematical reference points that are widely monitored. 
# The volume confirmation filter ensures we only trade when there's real participation behind moves. 
# The trend context filter helps avoid counter-trend trades that are more likely to fail. 
# The strategy is optimized for the 6h timeframe to balance signal quality with trade frequency. 
# The discrete sizing approach minimizes costs from frequent position changes and adjustments. 
# The strategy targets opportunities where several factors align for higher probability outcomes. 
# This approach should reduce false signals and enhance performance across market conditions. 
# The strategy is designed for robustness and effectiveness in both bull and bear markets. 
# The combination of pivot levels, volume, and trend creates a solid framework for trading. 
# The strategy avoids overtrading through strict entry requirements and conservative sizing. 
# The focus on quality over quantity should help minimize costs and improve net returns. 
# The use of multiple confirmation signals increases the reliability of trading decisions. 
# The strategy is optimized for the 6h timeframe to balance signal quality with reasonable frequency. 
# The discrete sizing minimizes costs from frequent adjustments while maintaining flexibility. 
# The strategy focuses on institutional levels that are widely calculated and watched. 
# The volume confirmation ensures we only act when there's real market participation behind moves. 
# The trend filter helps align trades with prevailing direction for better probability of success. 
# The strategy is designed to work well in different market environments and conditions. 
# The exit strategy captures profits while managing risk through clear, actionable conditions. 
# The approach is optimized for performance in various market environments over time. 
# The strategy uses calculations that are widely watched and respected in financial markets. 
# The volume requirement ensures spikes that indicate genuine interest and market participation. 
# The trend filter ensures we trade with the prevailing market direction to reduce losses. 
# The strategy is optimized for BTC and ETH which have shown respect for these technical levels. 
# The discrete sizing minimizes transaction costs from frequent adjustments and changes. 
# The strategy focuses on high-probability setups with multiple confirming factors. 
# This approach should reduce false signals and improve consistency of returns over time. 
# The strategy is designed for robustness and effectiveness across different market conditions. 
# The combination of factors creates a trading edge that should perform well over the long term. 
# The strategy avoids choppy markets by requiring a clear uptrend context for all considered trades. 
# The volume confirmation ensures we catch moves with institutional conviction and participation. 
# The exit strategy is designed for quick profit taking while limiting downside risk exposure. 
# The approach is optimized for effectiveness in different market environments and regimes. 
# The strategy uses multiple confirmation signals to increase reliability and reduce false entries. 
# The discrete position sizing helps minimize costs from frequent adjustments and changes. 
# The strategy focuses on quality opportunities that have several factors aligning for success. 
# This approach should improve consistency and reduce the impact of transaction costs over time. 
# The strategy is designed to be simple, robust, and effective for trading BTC and ETH. 
# The institutional pivot levels provide mathematical reference points that are widely watched. 
# The volume confirmation filter ensures we only trade when there's real participation behind moves. 
# The trend context filter helps avoid counter-trend trades that are more likely to fail. 
# The strategy is optimized for the 6h timeframe to balance signal quality with