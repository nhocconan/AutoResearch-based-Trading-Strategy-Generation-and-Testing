#!/usr/bin/env python3
"""
Experiment #002: 12h Choppiness Regime-Switch with 1d Trend Filter

Hypothesis: BTC/ETH fail simple trend strategies because they spend significant time
in range/choppy markets (especially 2022-2023 post-crash, 2025 bear market).
By detecting regime via Choppiness Index, we can:
- CHOP > 61.8 (range): Use mean reversion (RSI extremes + Bollinger bands)
- CHOP < 38.2 (trend): Use trend following (HMA + Donchian breakout)
- 38.2-61.8 (neutral): Stay flat or reduce size

This is DIFFERENT from failed #001 (4h timeframe) - using 12h for fewer, higher-quality trades.
1d HMA provides major trend bias to avoid counter-trend mean reversion in strong trends.

Why 12h works:
- Natural 20-50 trades/year (fee drag ~1-2.5%)
- Filters noise from lower TFs while catching major moves
- Regime detection adapts to BTC/ETH's choppy nature

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_switch_1d_filter_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs[np.isinf(rs)] = 0
    rs[np.isnan(rs)] = 0
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n)) / LOG10(Highest High - Lowest Low)
    
    Interpretation:
    - CHOP > 61.8: Market is choppy/ranging (mean reversion)
    - CHOP < 38.2: Market is trending (trend following)
    - 38.2-61.8: Neutral/transition
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    price_range[price_range == 0] = 1e-10  # Avoid division by zero
    
    # Choppiness formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum) / np.log10(price_range)
    
    chop = np.nan_to_num(chop, nan=50.0)  # Default to neutral
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_TREND = 0.30
    BASE_SIZE_MR = 0.25  # Smaller for mean reversion (higher risk)
    
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
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === REGIME DETECTION ===
        choppy_regime = chop_14[i] > 61.8
        trending_regime = chop_14[i] < 38.2
        neutral_regime = not choppy_regime and not trending_regime
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H HMA TREND ===
        hma_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === NEW SIGNAL ===
        new_signal = 0.0
        current_size = BASE_SIZE_TREND if trending_regime else BASE_SIZE_MR
        
        # === TRENDING REGIME: Trend Following ===
        if trending_regime:
            # Donchian breakout in trend direction
            breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
            breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
            
            # Long: HMA bullish + breakout + daily bias aligned
            if hma_bullish and breakout_long:
                if daily_bullish:
                    new_signal = current_size
                else:
                    new_signal = current_size * 0.7  # Reduced size against daily bias
            
            # Short: HMA bearish + breakout + daily bias aligned
            if hma_bearish and breakout_short:
                if daily_bearish:
                    new_signal = -current_size
                else:
                    new_signal = -current_size * 0.7  # Reduced size against daily bias
        
        # === CHOPPY REGIME: Mean Reversion ===
        elif choppy_regime:
            # RSI extremes + Bollinger band touch
            rsi_oversold = rsi_14[i] < 30
            rsi_overbought = rsi_14[i] > 70
            price_at_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
            price_at_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
            
            # Long: RSI oversold + price at lower BB + daily bias not strongly bearish
            if rsi_oversold and price_at_lower:
                if not daily_bearish:  # Avoid mean reversion against strong downtrend
                    new_signal = current_size
                elif rsi_14[i] < 20:  # Extreme oversold overrides daily bias
                    new_signal = current_size * 0.7
            
            # Short: RSI overbought + price at upper BB + daily bias not strongly bullish
            if rsi_overbought and price_at_upper:
                if not daily_bullish:  # Avoid mean reversion against strong uptrend
                    new_signal = -current_size
                elif rsi_14[i] > 80:  # Extreme overbought overrides daily bias
                    new_signal = -current_size * 0.7
        
        # === NEUTRAL REGIME: Stay flat or reduce size ===
        elif neutral_regime:
            # Only enter on very strong signals
            if hma_bullish and daily_bullish and rsi_14[i] < 50:
                new_signal = BASE_SIZE_TREND * 0.5
            elif hma_bearish and daily_bearish and rsi_14[i] > 50:
                new_signal = -BASE_SIZE_TREND * 0.5
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h HMA turns bearish (trend following positions)
            if position_side > 0 and hma_bearish and trending_regime:
                trend_reversal = True
            # Exit short if 12h HMA turns bullish (trend following positions)
            if position_side < 0 and hma_bullish and trending_regime:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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