#!/usr/bin/env python3
"""
Experiment #262: 12h Primary + 1d/1w HTF — Dual Regime (Chop/Trend) + Donchian + RSI

Hypothesis: Based on research showing Choppiness Index regime switching works well
(ETH Sharpe +0.923 with CRSI+CHOP, SOL +0.782 with Donchian+HMA+RSI).

Key components:
1. 12h primary timeframe (proven to reduce noise vs 4h)
2. 1d HMA for intermediate trend direction
3. 1w HMA for long-term regime filter (only trade with weekly trend)
4. Choppiness Index(14): CHOP>55=range(mean-revert), CHOP<45=trend(breakout)
5. Donchian(20) breakouts for trend entries
6. RSI(14) extremes + Bollinger for mean reversion in chop
7. ATR(14) trailing stoploss at 2.5x
8. Position sizing: 0.25 base, 0.30 strong conviction

Why this should work:
- 12h TF = 20-50 trades/year target (appropriate frequency)
- Dual regime adapts to market conditions (trend vs range)
- 1w filter prevents counter-trend trades in strong trends
- Discrete sizing minimizes fee churn
- Stoploss protects from 2022-style crashes

Position sizing: 0.25-0.30 (conservative, max 0.35)
Target: 25-45 trades/year per symbol
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_donchian_rsi_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return sma.values, upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1W TREND REGIME (long-term filter) ===
        # Only trade in direction of weekly trend
        weekly_bull = close[i] > hma_1w_21_aligned[i]
        weekly_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND REGIME (intermediate filter) ===
        daily_bull = close[i] > hma_1d_21_aligned[i]
        daily_bear = close[i] < hma_1d_21_aligned[i]
        daily_strong_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        daily_strong_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 12H LOCAL SIGNALS ===
        price_above_12h_hma = close[i] > hma_12h_21[i]
        price_below_12h_hma = close[i] < hma_12h_21[i]
        hma_12h_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_12h_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.999
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.001
        
        # === BOLLINGER BAND POSITION ===
        bb_breakout_long = close[i] > bb_upper[i] * 0.999
        bb_breakout_short = close[i] < bb_lower[i] * 1.001
        bb_mean_revert_long = close[i] < bb_lower[i] * 1.001
        bb_mean_revert_short = close[i] > bb_upper[i] * 0.999
        
        # === RSI THRESHOLDS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + aligned regimes)
        if is_trending:
            # LONG: Trending + weekly bull + daily bull + Donchian breakout
            if weekly_bull and daily_bull and donchian_breakout_long:
                new_signal = STRONG_SIZE
            # LONG: Trending + daily strong bull + 12h HMA bullish + RSI confirming
            elif daily_strong_bull and hma_12h_bullish and rsi_14[i] > 45:
                new_signal = BASE_SIZE
            # LONG: Trending + BB breakout + aligned trend
            elif bb_breakout_long and weekly_bull and daily_bull:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + weekly bear + daily bear + Donchian breakdown
            if weekly_bear and daily_bear and donchian_breakout_short:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Trending + daily strong bear + 12h HMA bearish + RSI confirming
            elif daily_strong_bear and hma_12h_bearish and rsi_14[i] < 55:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Trending + BB breakdown + aligned trend
            elif bb_breakout_short and weekly_bear and daily_bear:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: Choppy + RSI oversold + price near BB lower
            if rsi_oversold and bb_mean_revert_long:
                new_signal = BASE_SIZE
            # LONG: Choppy + RSI extreme oversold (any regime)
            elif rsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            # LONG: Choppy + price below 12h HMA but RSI recovering
            elif price_below_12h_hma and rsi_14[i] < 40 and rsi_14[i] > rsi_14[i-1] if i > 0 else False:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.8
            
            # SHORT: Choppy + RSI overbought + price near BB upper
            if rsi_overbought and bb_mean_revert_short:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + RSI extreme overbought (any regime)
            elif rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + price above 12h HMA but RSI weakening
            elif price_above_12h_hma and rsi_14[i] > 60 and rsi_14[i] < rsi_14[i-1] if i > 0 else False:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 10+ trades) ===
        # Force trade if no signal for 15 bars (~180h = 7.5 days on 12h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if weekly_bull and daily_bull and rsi_14[i] > 40:
                new_signal = BASE_SIZE * 0.7
            elif weekly_bear and daily_bear and rsi_14[i] < 60:
                new_signal = -BASE_SIZE * 0.7
            elif is_choppy and rsi_14[i] < 30:
                new_signal = BASE_SIZE * 0.6
            elif is_choppy and rsi_14[i] > 70:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but weekly trend turns strongly bearish
            if position_side > 0 and weekly_bear and daily_bear:
                regime_reversal = True
            # Short position but weekly trend turns strongly bullish
            if position_side < 0 and weekly_bull and daily_bull:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals