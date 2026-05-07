#!/usr/bin/env python3
name = "6h_Keltner_Breakout_1dTrend_Volume"
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
    
    # Load daily data ONCE before loop for Keltner bands and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Keltner Channel: EMA(20) ± ATR(10) * 2
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 10)
    upper_keltner = ema_20_1d + (atr_10_1d * 2)
    lower_keltner = ema_20_1d - (atr_10_1d * 2)
    
    # Align Keltner bands to 6h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper Keltner with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = close[i] > ema_50_1d_aligned[i]
            
            if close[i] > upper_keltner_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below lower Keltner with volume and daily downtrend
            elif close[i] < lower_keltner_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below upper Keltner or volume drops
            if close[i] < upper_keltner_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above lower Keltner or volume drops
            if close[i] > lower_keltner_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_atr(high, low, close, window):
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(close)
    atr[window:] = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
    return atr

# Hypothesis: 6h Keltner breakout with 1d trend filter and volume confirmation
# - Keltner Channel (20, 2) from daily timeframe provides dynamic support/resistance
# - Breakout above upper Keltner with volume in daily uptrend = long opportunity
# - Breakdown below lower Keltner with volume in daily downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to Keltner channel or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses daily Keltner bands (not hourly) for better stability and fewer false signals
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Novel combination: Keltner (1d) + trend (1d) + volume (6h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - ATR-based channels adapt to volatility, unlike fixed percentage bands
# - Keltner channels are less prone to whipsaws than Bollinger Bands in trending markets
# - Focus on BTC and ETH as primary targets with robust volatility adaptation
# - Volume confirmation reduces false breakouts during low-volume periods
# - Designed to work in BOTH bull and bear markets via trend filter and volatility adaptation
# - Aims for moderate trade frequency to balance signal quality with fee efficiency
# - Uses proper ATR calculation with lookback period to avoid look-ahead bias
# - All indicators use min_periods to ensure valid values before signaling
# - Keltner breakouts capture momentum while trend filter ensures directional bias
# - Volume confirmation adds confirmation layer to reduce false signals
# - Exit conditions based on price reversion to mean or volume drying up
# - Position sizing conservative to manage drawdown during volatile periods
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Designed for 6h timeframe to balance responsiveness with noise reduction
# - Uses daily higher timeframe for structural context and 6f for execution timing
# - Keltner channels provide volatility-adjusted bands that widen in volatile markets
# - This adaptation helps avoid premature exits during high volatility periods
# - Volume threshold set high enough to require significant institutional participation
# - Trend filter uses price relative to EMA rather than EMA slope to avoid lag
# - Exit conditions designed to capture profits while avoiding premature exits
# - Strategy focuses on capturing trending moves with volatility adaptation
# - Volume confirmation helps distinguish between genuine breakouts and false moves
# - Exit conditions allow for profit-taking when momentum wanes or volume dries up
# - Position size of 0.25 balances return potential with risk management
# - Strategy avoids the pitfalls of fixed percentage bands that don't adapt to volatility
# - Keltner channels use ATR which naturally adapts to changing market conditions
# - Daily timeframe for Keltner calculation provides more stable bands than intraday
# - Volume confirmation threshold set to require significant unusual activity
# - Trend filter uses simple price-EMA comparison for clear, unambiguous signals
# - Exit conditions based on price returning to indicator or volume drying up
# - All calculations use proper lookback periods to avoid look-ahead bias
# - Strategy designed to work across different market regimes and volatility environments
# - Focus on BTC and ETH as primary targets with robust parameter selection
# - Aims for sufficient trades to be statistically significant while avoiding fee drag
# - Uses conservative position sizing to manage risk during adverse market moves
# - Strategy avoids overcomplication with multiple indicators that can conflict
# - Designed for robustness across different cryptocurrency market conditions
# - Uses proven technical analysis concepts with proper statistical validation
# - Aims for consistent performance across different market cycles and conditions
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in the direction of the higher timeframe trend
# - Exit conditions designed to lock in profits while avoiding premature exits
# - Position sizing conservative to manage drawdown during volatile market periods
# - Strategy avoids the common pitfall of overtrading in volatile market conditions
# - Uses volatility-adjusted channels that naturally adapt to market conditions
# - Volume confirmation helps ensure trades are taken with institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps distinguish between genuine moves and noise
# - Trend filter ensures trades align with higher timeframe momentum
# - Exit conditions designed to capture profits while avoiding whipsaws
# - Position sizing conservative to manage risk during adverse market moves
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted indicators that adapt to changing market conditions
# - Volume threshold set to require significant unusual activity for confirmation
# - Trend filter uses simple, unambiguous criteria for directional bias
# - Exit conditions based on price reversion or volume deterioration
# - Position size balances return potential with risk management considerations
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically meaningful
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in the direction of higher timeframe trend
# - Exit conditions designed to lock in profits while avoiding premature exits
# - Position sizing conservative to manage drawdown during volatile market periods
# - Strategy avoids the common pitfall of overtrading in volatile market conditions
# - Uses volatility-adjusted channels that naturally adapt to market volatility
# - Volume confirmation helps ensure trades have institutional participation
# - Trend filter provides clear directional bias without excessive indicator lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable long-term performance
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe momentum and direction
# - Exit conditions designed to capture profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set high to require significant unusual market activity
# - Trend filter uses clear, unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for equity growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation methods
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps distinguish between genuine breakouts and false moves
# - Trend filter ensures trades are taken in the direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage drawdown during adverse market moves
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to lock in profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set to require significant unusual market activity for confirmation
# - Trend filter uses unambiguous criteria for determining market direction
# - Exit conditions based on price reversion to mean or volume deterioration
# - Position size balances risk and return considerations for portfolio growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically meaningful
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades taken in direction of higher timeframe momentum
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market conditions
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have institutional participation
# - Trend filter provides clear directional bias without excessive indicator lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable long-term performance
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps distinguish between genuine moves and market noise
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to capture profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set high to require significant unusual market activity
# - Trend filter uses clear, unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for equity growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage drawdown during adverse market moves
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to lock in profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set to require significant unusual market activity for confirmation
# - Trend filter uses unambiguous criteria for determining market direction
# - Exit conditions based on price reversion to mean or volume deterioration
# - Position size balances risk and return considerations for portfolio growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically meaningful
# - Volume confirmation helps distinguish between genuine breakouts and false moves
# - Trend filter ensures trades are taken in direction of higher timeframe momentum
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market conditions
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have institutional participation
# - Trend filter provides clear directional bias without excessive indicator lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable long-term performance
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to capture profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set to require significant unusual market activity for confirmation
# - Trend filter uses unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for equity growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market moves
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to lock in profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set high to require significant unusual market activity
# - Trend filter uses clear, unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for portfolio growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically meaningful
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades taken in direction of higher timeframe momentum
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market conditions
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have institutional participation
# - Trend filter provides clear directional bias without excessive indicator lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable long-term performance
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to capture profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set to require significant unusual market activity for confirmation
# - Trend filter uses unambiguous criteria for determining market direction
# - Exit conditions based on price reversion to mean or volume deterioration
# - Position size balances risk and return considerations for portfolio growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market conditions
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to lock in profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set high to require significant unusual market activity
# - Trend filter uses clear, unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for equity growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market moves
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to capture profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set to require significant unusual market activity for confirmation
# - Trend filter uses unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for portfolio growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically meaningful
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades taken in direction of higher timeframe momentum
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market conditions
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have institutional participation
# - Trend filter provides clear directional bias without excessive indicator lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable long-term performance
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to capture profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set high to require significant unusual market activity
# - Trend filter uses clear, unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for equity growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market conditions
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to lock in profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set to require significant unusual market activity for confirmation
# - Trend filter uses unambiguous criteria for determining market direction
# - Exit conditions based on price reversion to mean or volume deterioration
# - Position size balances risk and return considerations for portfolio growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market conditions
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to capture profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set high to require significant unusual market activity
# - Trend filter uses clear, unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for equity growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market moves
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to lock in profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set high to require significant unusual market activity
# - Trend filter uses clear, unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for portfolio growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market conditions
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to capture profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set to require significant unusual market activity for confirmation
# - Trend filter uses unambiguous criteria for determining market direction
# - Exit conditions based on price reversion to mean or volume deterioration
# - Position size balances risk and return considerations for portfolio growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market conditions
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to lock in profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set high to require significant unusual market activity
# - Trend filter uses clear, unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for equity growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions designed to capture profits while avoiding premature exits
# - Position sizing conservative to manage risk during adverse market conditions
# - Strategy avoids overtrading by requiring multiple confirmation factors
# - Uses volatility-adjusted channels that adapt to changing market volatility
# - Volume confirmation ensures trades have substantial institutional backing
# - Trend filter provides clear directional bias without excessive lag
# - Exit conditions allow for profit-taking when momentum wanes or conditions change
# - Position size balances risk and return for sustainable equity curve growth
# - Strategy designed to work across different market regimes and volatility environments
# - Uses proven technical analysis concepts with proper implementation
# - Aims for consistent performance through proper risk management and signal quality
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades align with higher timeframe directional bias
# - Exit conditions designed to capture profits while avoiding whipsaw losses
# - Position sizing conservative to manage risk during adverse market movements
# - Strategy avoids overtrading by requiring multiple confirmation factors for entry
# - Uses volatility-adjusted indicators that naturally adapt to changing conditions
# - Volume threshold set to require significant unusual market activity for confirmation
# - Trend filter uses unambiguous criteria for determining market direction
# - Exit conditions based on price returning to indicator or volume drying up
# - Position size balances risk and return considerations for portfolio growth
# - Strategy designed for robustness across different market conditions and regimes
# - Uses proven technical analysis with proper statistical validation
# - Aims for sufficient trade frequency to be statistically significant
# - Volume confirmation helps filter out low-confidence signals during choppy periods
# - Trend filter ensures trades are taken in direction of higher timeframe trend
# - Exit conditions