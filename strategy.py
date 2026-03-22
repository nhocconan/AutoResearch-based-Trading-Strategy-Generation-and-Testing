#!/usr/bin/env python3
"""
Experiment #002: 30m Regime-Adaptive Strategy with Choppiness Index + 4h HMA Bias
Hypothesis: 30m timeframe balances noise reduction with trade frequency. Choppiness Index
detects range vs trend regimes, allowing adaptive strategy selection. In range markets (CHOP>61.8),
use mean reversion (RSI extremes + Bollinger bands). In trend markets (CHOP<38.2), use trend
following (HMA/EMA alignment). 4h HMA provides HTF bias filter. Multiple entry paths ensure
>=10 trades per symbol. Conservative sizing (0.25-0.30) controls drawdown. 2.0*ATR stoploss.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_regime_4h_hma_rsi_bb_atr_v1"
timeframe = "30m"
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
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_keltner_channels(high, low, close, period=20, atr_period=10, mult=2.0):
    """Calculate Keltner Channels for trend confirmation."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper, lower, ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    chop = calculate_choppiness_index(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    kc_upper, kc_lower, kc_ema = calculate_keltner_channels(high, low, close, 20, 10, 2.0)
    
    # HMA for trend direction
    hma_30 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Regime detection via Choppiness Index
        regime_range = chop[i] > 55  # Range/choppy market
        regime_trend = chop[i] < 45  # Trending market
        
        # 30m trend indicators
        hma_bullish = hma_30[i] > hma_50[i] if not np.isnan(hma_50[i]) else False
        hma_bearish = hma_30[i] < hma_50[i] if not np.isnan(hma_50[i]) else False
        ema_bullish = ema_21[i] > ema_50[i] if not np.isnan(ema_50[i]) else False
        ema_bearish = ema_21[i] < ema_50[i] if not np.isnan(ema_50[i]) else False
        
        # RSI zones
        rsi_oversold = rsi[i] < 32
        rsi_overbought = rsi[i] > 68
        rsi_neutral_long = rsi[i] > 42 and rsi[i] < 58
        rsi_neutral_short = rsi[i] > 42 and rsi[i] < 58
        
        # Bollinger Band positions
        price_near_bb_lower = close[i] < bb_lower[i] * 1.005
        price_near_bb_upper = close[i] > bb_upper[i] * 0.995
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_sma[i] < 0.10
        
        # Keltner Channel positions
        price_below_kc = close[i] < kc_lower[i]
        price_above_kc = close[i] > kc_upper[i]
        
        # EMA/HMA crossover signals
        hma_cross_long = hma_30[i] > hma_50[i] and hma_30[i-1] <= hma_50[i-1] if i > 0 and not np.isnan(hma_50[i-1]) else False
        hma_cross_short = hma_30[i] < hma_50[i] and hma_30[i-1] >= hma_50[i-1] if i > 0 and not np.isnan(hma_50[i-1]) else False
        
        new_signal = 0.0
        
        # === RANGE REGIME (Mean Reversion) ===
        if regime_range:
            # Long: RSI oversold + price near BB lower + 4h not bearish
            if rsi_oversold and price_near_bb_lower and not hma_4h_bearish:
                new_signal = SIZE_ENTRY
            
            # Short: RSI overbought + price near BB upper + 4h not bullish
            elif rsi_overbought and price_near_bb_upper and not hma_4h_bullish:
                new_signal = -SIZE_ENTRY
            
            # Long: Price below Keltner lower + RSI < 40
            elif price_below_kc and rsi[i] < 40 and not hma_4h_bearish:
                new_signal = SIZE_ENTRY
            
            # Short: Price above Keltner upper + RSI > 60
            elif price_above_kc and rsi[i] > 60 and not hma_4h_bullish:
                new_signal = -SIZE_ENTRY
        
        # === TREND REGIME (Trend Following) ===
        if regime_trend:
            # Long: 4h HMA bullish + 30m HMA bullish + RSI neutral pullback
            if hma_4h_bullish and hma_bullish and rsi_neutral_long:
                new_signal = SIZE_ENTRY
            
            # Short: 4h HMA bearish + 30m HMA bearish + RSI neutral pullback
            elif hma_4h_bearish and hma_bearish and rsi_neutral_short:
                new_signal = -SIZE_ENTRY
            
            # Long: HMA cross long + 4h bullish confirmation
            elif hma_cross_long and hma_4h_bullish:
                new_signal = SIZE_ENTRY
            
            # Short: HMA cross short + 4h bearish confirmation
            elif hma_cross_short and hma_4h_bearish:
                new_signal = -SIZE_ENTRY
            
            # Long: EMA bullish + price above KC middle + RSI > 50
            elif ema_bullish and close[i] > kc_ema[i] and rsi[i] > 50 and hma_4h_bullish:
                new_signal = SIZE_ENTRY
            
            # Short: EMA bearish + price below KC middle + RSI < 50
            elif ema_bearish and close[i] < kc_ema[i] and rsi[i] < 50 and hma_4h_bearish:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME (CHOP 45-55) - Reduced sizing ===
        if not regime_range and not regime_trend:
            # Only take high-confidence setups with reduced size
            if rsi_oversold and price_near_bb_lower and hma_4h_bullish:
                new_signal = SIZE_ENTRY * 0.7
            elif rsi_overbought and price_near_bb_upper and hma_4h_bearish:
                new_signal = -SIZE_ENTRY * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 30m timeframe)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 30m timeframe)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals