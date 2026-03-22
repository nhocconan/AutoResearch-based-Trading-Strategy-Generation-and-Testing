#!/usr/bin/env python3
"""
Experiment #095: 1h Primary + 4h/1d HTF — Volume-Weighted RSI + Choppiness Regime + Session Filter

Hypothesis: Lower timeframe (1h) strategies fail due to excessive trade frequency causing fee drag.
This strategy uses 4h Choppiness for regime detection (not 1h), 1d HMA for major trend bias,
and only enters on 1h when ALL filters align. Session filter (8-20 UTC) + volume filter +
strict RSI thresholds should limit trades to 40-80/year while maintaining edge.

Strategy Logic:
1. 1d HMA(21) SLOPE: Major trend bias (only longs if slope > 0, shorts if < 0)
2. 4h CHOPPINESS (14): CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
3. 1h RSI(14) + Volume: Entry timing with volume confirmation (>0.8x 20-bar avg)
4. SESSION FILTER: Only trade 8-20 UTC (reduces trades by ~50%, avoids Asia chop)
5. ATR(14) stoploss: 2.5x trailing stop
6. Position size: 0.25 discrete (conservative for 1h TF)

Why this should work:
- 4h regime detection is more stable than 1h (fewer false signals)
- 1d trend bias prevents counter-trend trades in major moves
- Session filter eliminates low-liquidity periods (Asia session = whipsaw)
- Volume filter ensures entries have conviction
- Strict RSI thresholds (20/80 for range, 35/65 for trend) reduce trade count
- 1h timeframe allows precise entry timing within HTF trend

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h (regime) + 1d (trend bias) via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vw_rsi_chop_session_4h1d_v1"
timeframe = "1h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)  # avoid div by zero
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (regime detection)
    chop_4h_14 = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        14
    )
    
    # Calculate 1d indicators (trend bias)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_14)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 1h HMA for trend confirmation
    hma_1h_21 = calculate_hma(close, 21)
    hma_1h_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    # Lower TF = smaller size to reduce fee impact
    BASE_SIZE = 0.25
    
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
        
        if np.isnan(chop_4h_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        if np.isnan(hma_1h_21[i]) or np.isnan(hma_1h_50[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Reduces trades by ~50%, avoids Asia session chop
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === 1D TREND BIAS (MAJOR) ===
        # HMA slope > 0.5 = bullish bias (prefer longs)
        # HMA slope < -0.5 = bearish bias (prefer shorts)
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        
        # Price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = range market (mean revert strategy)
        # CHOP < 45 = trend market (trend follow strategy)
        # CHOP between = transitional (use weaker signals or skip)
        is_range_market = chop_4h_aligned[i] > 55
        is_trend_market = chop_4h_aligned[i] < 45
        
        # === 1H RSI SIGNALS ===
        # Range market: extreme RSI for mean reversion (20/80)
        # Trend market: moderate RSI for pullback entries (35/65)
        rsi_oversold = rsi_14[i] < 20
        rsi_overbought = rsi_14[i] > 80
        rsi_moderate_low = rsi_14[i] < 35
        rsi_moderate_high = rsi_14[i] > 65
        
        # === 1H TREND CONFIRMATION ===
        hma_bullish = hma_1h_21[i] > hma_1h_50[i]
        hma_bearish = hma_1h_21[i] < hma_1h_50[i]
        
        # === VOLUME FILTER ===
        # Only enter if volume > 0.8x 20-bar average (ensures liquidity)
        volume_confirmed = vol_ratio[i] > 0.8
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transitional markets or outside session
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.5
        if not in_session:
            current_size = BASE_SIZE * 0.3
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if in_session and volume_confirmed:
            if is_range_market:
                # Mean reversion in range: buy extreme oversold
                # Requires: RSI < 20 + 1d bias not strongly bearish
                if rsi_oversold and not trend_1d_bearish:
                    new_signal = current_size
            elif is_trend_market:
                # Trend following: buy pullback in uptrend
                # Requires: 1d bullish + 1h bullish + RSI pullback
                if trend_1d_bullish and hma_bullish and rsi_moderate_low:
                    new_signal = current_size
                # Also allow if price above 1d HMA even without slope
                elif price_above_1d_hma and hma_bullish and rsi_14[i] < 40:
                    new_signal = current_size * 0.8
            else:
                # Transitional: weaker signals only with strong 1d bias
                if trend_1d_bullish and rsi_14[i] < 30:
                    new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        if in_session and volume_confirmed:
            if is_range_market:
                # Mean reversion in range: sell extreme overbought
                # Requires: RSI > 80 + 1d bias not strongly bullish
                if rsi_overbought and not trend_1d_bullish:
                    new_signal = -current_size
            elif is_trend_market:
                # Trend following: sell pullback in downtrend
                # Requires: 1d bearish + 1h bearish + RSI pullback
                if trend_1d_bearish and hma_bearish and rsi_moderate_high:
                    new_signal = -current_size
                # Also allow if price below 1d HMA even without slope
                elif price_below_1d_hma and hma_bearish and rsi_14[i] > 60:
                    new_signal = -current_size * 0.8
            else:
                # Transitional: weaker signals only with strong 1d bias
                if trend_1d_bearish and rsi_14[i] > 70:
                    new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~8 days on 1h), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if in_session and volume_confirmed:
                if trend_1d_bullish and rsi_14[i] < 35:
                    new_signal = current_size * 0.4
                elif trend_1d_bearish and rsi_14[i] > 65:
                    new_signal = -current_size * 0.4
        
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
        # Exit if regime changes strongly against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if market becomes strongly trending bearish
            if position_side > 0 and is_trend_market and trend_1d_bearish:
                regime_reversal = True
            # Exit short if market becomes strongly trending bullish
            if position_side < 0 and is_trend_market and trend_1d_bullish:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
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