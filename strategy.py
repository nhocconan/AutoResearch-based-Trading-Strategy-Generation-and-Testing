#!/usr/bin/env python3
"""
Experiment #011: 4h Volatility Regime Adaptive Strategy with 1d/1w Filters

Hypothesis: Previous strategies failed because they used static entry rules regardless
of market regime. This strategy ADAPTS based on volatility regime detected by ATR ratio:

REGIME 1 - VOLATILITY SPIKE (ATR7/ATR30 > 2.0):
- Market is in panic/euphoria (like 2022 crash bottoms, 2021 tops)
- Use MEAN REVERSION: fade extremes at Bollinger Band bounds
- Z-score filter confirms extreme moves
- Higher win rate, quicker exits when z-score normalizes

REGIME 2 - NORMAL VOLATILITY (ATR7/ATR30 <= 2.0):
- Market is trending or ranging normally
- Use TREND FOLLOWING: trade with 1d HMA direction
- ADX filter confirms trend strength
- Larger moves, longer holds

HTF FILTERS:
- 1d HMA(21) for major trend bias (only trade with daily trend)
- 1w HMA(21) for secular trend (reduce size against weekly trend)

Why this should work better than #004:
- Mean reversion during vol spikes captures 2022 crash bottoms (where trend following failed)
- Trend following during normal vol captures 2021 bull run
- HTF filters prevent counter-trend trades (major issue in 2022-2023)
- Regime-adaptive = different logic for different market conditions
- 4h timeframe targets 20-50 trades/year (manageable fee drag)

Timeframe: 4h (REQUIRED)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_regime_adaptive_1d_1w_filter_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average - smoother and more responsive than EMA.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reference: Alan Hull, 2005
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Weighted Moving Average helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close_s, half_period)
    wma_full = wma(close_s, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to moving average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std
    zscore = zscore.fillna(0).values
    return zscore

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = ranging market.
    Reference: Wilder, "New Concepts in Technical Trading Systems"
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values using Wilder's method (EMA with span=period)
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMA for trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    zscore_20 = calculate_zscore(close, 20)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # Volume moving average
    volume_s = pd.Series(volume)
    volume_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio for regime detection
    atr_ratio = np.divide(atr_7, atr_30, out=np.ones_like(atr_7), where=atr_30!=0)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(zscore_20[i]):
            continue
        
        if np.isnan(volume_ma20[i]) or volume_ma20[i] == 0:
            continue
        
        # === HTF TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === VOLATILITY REGIME ===
        vol_spike = atr_ratio[i] > 2.0  # High volatility - mean reversion
        vol_normal = atr_ratio[i] <= 2.0  # Normal volatility - trend follow
        
        # === ENTRY LOGIC BY REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # REGIME 1: VOLATILITY SPIKE - Mean Reversion
        if vol_spike:
            # Long: price at lower BB + extreme negative z-score + daily bullish bias
            if close[i] <= bb_lower[i] and zscore_20[i] < -1.8 and daily_bullish:
                new_signal = BASE_SIZE
            # Short: price at upper BB + extreme positive z-score + daily bearish bias
            elif close[i] >= bb_upper[i] and zscore_20[i] > 1.8 and daily_bearish:
                new_signal = -BASE_SIZE
        
        # REGIME 2: NORMAL VOLATILITY - Trend Following
        elif vol_normal:
            adx_strong = adx_14[i] > 20
            
            # Long: daily bullish + ADX confirms trend
            if daily_bullish and adx_strong:
                new_signal = BASE_SIZE
            # Short: daily bearish + ADX confirms trend
            elif daily_bearish and adx_strong:
                new_signal = -BASE_SIZE
        
        # === WEEKLY TREND SIZE ADJUSTMENT ===
        # Reduce size when trading against weekly trend (secular bias)
        if new_signal > 0 and weekly_bearish:
            new_signal = new_signal * 0.5  # Half size against weekly
        elif new_signal < 0 and weekly_bullish:
            new_signal = new_signal * 0.5  # Half size against weekly
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~17 days on 4h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if daily_bullish and weekly_bullish:
                new_signal = BASE_SIZE * 0.5
            elif daily_bearish and weekly_bearish:
                new_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === MEAN REVERSION TAKE PROFIT ===
        # Exit when z-score normalizes after vol spike entry
        mr_take_profit = False
        if in_position and position_side != 0:
            if position_side > 0 and zscore_20[i] > 0.5:
                mr_take_profit = True  # Long: z-score crossed above mean
            elif position_side < 0 and zscore_20[i] < -0.5:
                mr_take_profit = True  # Short: z-score crossed below mean
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if daily turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if daily turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or mr_take_profit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals