#!/usr/bin/env python3
"""
Experiment #317: 12h Regime-Adaptive with Choppiness Index + HTF HMA + Funding Contrarian

Hypothesis: Based on #316 success (Sharpe=0.676 on 4h with regime chop), adapt to 12h
with enhanced regime detection and funding rate contrarian edge for BTC/ETH.

Key insights from research:
1. Choppiness Index (CHOP) distinguishes range (CHOP>61.8) vs trend (CHOP<38.2)
2. In range regimes: mean revert at Bollinger bounds (RSI extremes)
3. In trend regimes: follow HTF HMA bias with EMA confirmation
4. Funding rate contrarian: short when funding>0.03%, long when <-0.03% (BTC/ETH edge)
5. 12h needs LOOSE conditions for >=10 trades but fewer signal changes for fee efficiency

Strategy logic:
1. CHOP(14) regime detection on 12h
2. 1d HMA(21) for primary trend bias (proven edge)
3. 1w HMA(21) for meta-trend confirmation
4. Bollinger Bands(20,2.0) for mean reversion entries in range regime
5. EMA(8/21) for trend entries in trend regime
6. ATR(14) trailing stoploss at 2.5x (proven from successful strategies)
7. Position sizing: 0.20-0.30 discrete, reduce in high volatility

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_chop_htf_hma_funding_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n) * np.nan
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range/choppy (mean reversion)
        # CHOP < 38.2 = trending (trend following)
        # 38.2 <= CHOP <= 61.8 = transition (stay flat or reduce size)
        range_regime = chop[i] > 61.8
        trend_regime = chop[i] < 38.2
        transition_regime = not range_regime and not trend_regime
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = primary directional bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA = meta-trend confirmation
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY ADJUSTMENT ===
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on regime and volatility
        if high_volatility or transition_regime:
            position_size = SIZE_BASE
        elif trend_regime and bull_trend_1w:
            position_size = SIZE_STRONG
        elif range_regime:
            position_size = SIZE_BASE  # mean reversion = smaller size
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # TREND REGIME: Follow HTF bias with EMA confirmation
        if trend_regime:
            # LONG: 1d bias up + EMA bullish + price above BB mid
            long_trend = (
                bull_trend_1d and
                ema_fast[i] > ema_slow[i] and
                close[i] > bb_mid[i]
            )
            
            # SHORT: 1d bias down + EMA bearish + price below BB mid
            short_trend = (
                bear_trend_1d and
                ema_fast[i] < ema_slow[i] and
                close[i] < bb_mid[i]
            )
            
            if long_trend:
                new_signal = position_size
            elif short_trend:
                new_signal = -position_size
        
        # RANGE REGIME: Mean reversion at Bollinger bounds with RSI extremes
        elif range_regime:
            # LONG: Price at lower BB + RSI oversold + 1d bias not strongly bearish
            long_range = (
                close[i] <= bb_lower[i] * 1.002 and  # at or below lower band
                rsi[i] < 35 and  # oversold
                not bear_trend_1w  # 1w not strongly bearish
            )
            
            # SHORT: Price at upper BB + RSI overbought + 1d bias not strongly bullish
            short_range = (
                close[i] >= bb_upper[i] * 0.998 and  # at or above upper band
                rsi[i] > 65 and  # overbought
                not bull_trend_1w  # 1w not strongly bullish
            )
            
            if long_range:
                new_signal = position_size
            elif short_range:
                new_signal = -position_size
        
        # === DONCHIAN BREAKOUT BOOST (in trend regime) ===
        # If price breaks Donchian high in trend regime + bull bias = stronger long
        if trend_regime and bull_trend_1d:
            if close[i] > donchian_upper[i] * 0.998:
                if new_signal > 0:
                    new_signal = SIZE_STRONG  # boost to max size
        
        # If price breaks Donchian low in trend regime + bear bias = stronger short
        if trend_regime and bear_trend_1d:
            if close[i] < donchian_lower[i] * 1.002:
                if new_signal < 0:
                    new_signal = -SIZE_STRONG  # boost to max size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit long if regime switches from trend to range while in long
        if in_position and new_signal != 0.0:
            if position_side > 0 and range_regime and close[i] > bb_mid[i]:
                # In range regime, exit if price is above mid (take profit)
                new_signal = 0.0
            if position_side < 0 and range_regime and close[i] < bb_mid[i]:
                # In range regime, exit if price is below mid (take profit)
                new_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d and not range_regime:
                # Exit long if 1d trend flips bearish (in trend regime)
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d and not range_regime:
                # Exit short if 1d trend flips bullish (in trend regime)
                new_signal = 0.0
        
        # === EMA REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and ema_fast[i] < ema_slow[i]:
                new_signal = 0.0
            if position_side < 0 and ema_fast[i] > ema_slow[i]:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # If same side, keep tracking for stoploss
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals