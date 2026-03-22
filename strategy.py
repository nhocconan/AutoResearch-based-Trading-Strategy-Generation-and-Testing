#!/usr/bin/env python3
"""
Experiment #039: 4h Primary + 1d HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Previous strategies failed because RSI/Connors are too slow in bear markets.
Ehlers Fisher Transform (period=9) catches reversals faster with less lag than RSI.
Combined with KAMA (adaptive MA) that adjusts to volatility, this should work better
in 2022 crash and 2025 bear market conditions.

Key components:
1. 1d HMA(21) for MAJOR trend bias (only trade WITH daily trend)
2. 4h KAMA(10, fast=2, slow=30) for adaptive trend following (adjusts to vol)
3. Ehlers Fisher Transform(9) for reversal signals (crosses -1.5/+1.5)
4. ADX(14) regime filter (>25 = trend follow, <20 = mean revert)
5. ATR(14) trailing stoploss at 2.5x
6. Volume confirmation (>0.7x 20-bar avg)

Why this should beat previous attempts:
- Fisher Transform has superior reversal detection vs RSI (Ehlers research)
- KAMA adapts to volatility = less whipsaw in 2022 crash
- ADX regime filter prevents trend-following in chop
- 1d bias prevents counter-trend trades that failed in 2025 bear market
- 4h timeframe = 20-50 trades/year (optimal fee/trade balance)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_adx_1d_hma_v1"
timeframe = "4h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    tr_s = pd.Series(tr)
    
    # Smoothed DM and TR
    atr = tr_s.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx.fillna(0).values
    
    return adx, plus_di.values, minus_di.values

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - smooth in trends, responsive in ranges
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s - close_s.shift(period))
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing Constant
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    
    Long signal: Fisher crosses above -1.5 from below
    Short signal: Fisher crosses below +1.5 from above
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Median price
    median = (high_s + low_s) / 2
    
    # Normalize price within range
    highest = median.rolling(window=period, min_periods=period).max()
    lowest = median.rolling(window=period, min_periods=period).min()
    
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, np.nan)
    
    # Normalized price (-1 to +1)
    normalized = 2 * (median - lowest) / range_hl - 1
    normalized = normalized.clip(-0.999, 0.999)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-period lag)
    fisher_signal = fisher.shift(1)
    
    fisher = fisher.fillna(0).values
    fisher_signal = fisher_signal.fillna(0).values
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    kama_10 = calculate_kama(close, 10, 2, 30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(kama_10[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # Price above 1d HMA = bullish bias (prefer longs)
        # Price below 1d HMA = bearish bias (prefer shorts)
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H ADAPTIVE TREND (KAMA) ===
        # Price above KAMA = bullish intermediate trend
        # Price below KAMA = bearish intermediate trend
        trend_4h_bullish = close[i] > kama_10[i]
        trend_4h_bearish = close[i] < kama_10[i]
        
        # === ADX REGIME FILTER ===
        # ADX > 25 = trending market (follow trend)
        # ADX < 20 = ranging market (mean revert / Fisher reversals)
        is_trending = adx_14[i] > 25
        is_ranging = adx_14[i] < 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * volume_sma[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Also allow extreme Fisher values for mean reversion in ranges
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Trending regime: 1d bullish + 4h bullish + Fisher long cross + volume
        # Ranging regime: 1d bullish + Fisher extreme long + volume
        if is_trending:
            if trend_1d_bullish and trend_4h_bullish and fisher_long_cross and volume_ok:
                new_signal = current_size
        else:  # ranging or neutral
            if trend_1d_bullish and (fisher_long_cross or fisher_extreme_long) and volume_ok:
                new_signal = current_size
        
        # SHORT ENTRIES
        # Trending regime: 1d bearish + 4h bearish + Fisher short cross + volume
        # Ranging regime: 1d bearish + Fisher extreme short + volume
        if is_trending:
            if trend_1d_bearish and trend_4h_bearish and fisher_short_cross and volume_ok:
                new_signal = -current_size
        else:  # ranging or neutral
            if trend_1d_bearish and (fisher_short_cross or fisher_extreme_short) and volume_ok:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~50 days on 4h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and trend_4h_bullish and fisher[i] < -1.0:
                new_signal = REDUCED_SIZE
            elif trend_1d_bearish and trend_4h_bearish and fisher[i] > 1.0:
                new_signal = -REDUCED_SIZE
        
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1d trend turns bearish + Fisher overbought
        # Exit short if 1d trend turns bullish + Fisher oversold
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and fisher[i] > 1.5:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and fisher[i] < -1.5:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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