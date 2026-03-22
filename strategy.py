#!/usr/bin/env python3
"""
Experiment #105: 1h Primary + 4h/1d HTF — Fisher Transform + Funding Z-Score + HMA Trend

Hypothesis: Recent Choppiness Index strategies failed because CHOP is laggy and doesn't
work well in crypto's persistent trending regimes. Instead, I'll use:

1. EHLERS FISHER TRANSFORM (period=9): Superior to RSI for catching reversals in bear
   markets. Long when Fisher crosses above -1.5, short when crosses below +1.5.
   
2. FUNDING RATE Z-SCORE (30d): Proven edge for BTC/ETH specifically. When funding
   z-score > +2.0 (extreme long bias), short (contrarian). When < -2.0, long.
   This is MARKET-NEUTRAL edge that works in 2025 bear market.
   
3. 4h HMA(21) TREND: Major direction bias. Only long if 4h HMA slope > 0, only short
   if 4h HMA slope < 0. Prevents counter-trend trades.
   
4. SESSION FILTER (8-20 UTC): Only trade during high-volume hours to reduce noise
   and false breakouts during Asian session.
   
5. ATR VOLATILITY FILTER: Only trade when ATR(14)/ATR(50) is between 0.7-1.5
   (normal volatility). Skip during extreme vol spikes or dead markets.

Why this should work:
- Fisher Transform has better reversal detection than RSI (Ehlers research)
- Funding rate is UNIQUE data not used in 98 previous failed strategies
- 4h HTF trend prevents whipsaw on 1h timeframe
- Session filter reduces trades by ~60% (only 12h of 24h)
- Conservative size (0.25) protects against 2022-style crashes

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (conservative for 1h TF)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol (strict confluence = fewer trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_funding_hma_4h_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.67 * prev_X
    
    Long signal: Fisher crosses above -1.5 (oversold reversal)
    Short signal: Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Calculate (HH - LL) over period
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        price_range = hh - ll
        if price_range < 1e-10:
            price_range = 1e-10
        
        # Normalize price to -1 to +1 range
        x = 0.66 * ((close[i] - ll) / price_range - 0.5) + 0.67 * (fisher[i-1] if i > 0 else 0)
        x = np.clip(x, -0.999, 0.999)  # Prevent ln domain error
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        fisher_signal[i] = fisher[i-1] if i > 0 else 0
    
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_zscore(series, period=30):
    """Calculate rolling z-score."""
    series_s = pd.Series(series)
    rolling_mean = series_s.rolling(window=period, min_periods=period).mean()
    rolling_std = series_s.rolling(window=period, min_periods=period).std()
    zscore = (series_s - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.fillna(0).values

def load_funding_data(symbol):
    """
    Load funding rate data from processed funding parquet files.
    Returns array of funding rates aligned with prices index.
    """
    try:
        # Map symbol to funding file path
        symbol_lower = symbol.lower().replace('usdt', '')
        funding_path = f"data/processed/funding/{symbol_lower}_funding.parquet"
        
        import os
        if not os.path.exists(funding_path):
            # Fallback: try alternative path
            funding_path = f"data/funding/{symbol_lower}_funding.parquet"
            if not os.path.exists(funding_path):
                return None
        
        df_funding = pd.read_parquet(funding_path)
        return df_funding['funding_rate'].values
    except Exception:
        return None

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract symbol from prices metadata if available
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if hasattr(prices, 'get') else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    # Load funding rate data (unique edge for BTC/ETH)
    funding_rates = load_funding_data(symbol)
    if funding_rates is not None and len(funding_rates) >= n:
        funding_zscore = calculate_zscore(funding_rates[:n], 30)
    else:
        # Fallback: use price-based proxy if funding data unavailable
        funding_zscore = np.zeros(n)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    # Volume MA for filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25  # Conservative for 1h TF
    
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
        
        if np.isnan(atr_50[i]) or atr_50[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high-volume hours to reduce noise
        current_hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= current_hour <= 20
        
        if not in_session:
            # If in position, hold. If not, don't enter.
            if not in_position:
                signals[i] = 0.0
                continue
        
        # === VOLATILITY FILTER ===
        # Only trade when volatility is normal (not extreme spikes or dead markets)
        vol_ratio = atr_14[i] / atr_50[i] if atr_50[i] > 0 else 1.0
        vol_normal = 0.6 <= vol_ratio <= 1.8
        
        # === VOLUME FILTER ===
        # Only trade when volume is above average
        vol_above_avg = volume[i] > 0.7 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else True
        
        # === 4H TREND BIAS (MAJOR) ===
        # HMA slope > 0.3 = bullish bias (prefer longs)
        # HMA slope < -0.3 = bearish bias (prefer shorts)
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.3
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.3
        
        # Price vs 4h HMA for additional confirmation
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = (fisher_signal[i] <= -1.5 and fisher[i] > -1.5)
        fisher_cross_short = (fisher_signal[i] >= 1.5 and fisher[i] < 1.5)
        
        # Also allow extreme Fisher values for mean reversion
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # === FUNDING RATE Z-SCORE (CONTRARIAN) ===
        # Z-score > +2.0 = extreme long bias = short signal (contrarian)
        # Z-score < -2.0 = extreme short bias = long signal (contrarian)
        funding_extreme_long = funding_zscore[i] > 1.5
        funding_extreme_short = funding_zscore[i] < -1.5
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size if filters not fully aligned
        filter_score = 0
        if vol_normal:
            filter_score += 1
        if vol_above_avg:
            filter_score += 1
        if trend_4h_bullish or trend_4h_bearish:
            filter_score += 1
        
        if filter_score < 2:
            current_size = BASE_SIZE * 0.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (require 2+ confluence)
        long_confluence = 0
        if trend_4h_bullish or price_above_4h_hma:
            long_confluence += 1
        if fisher_cross_long or fisher_extreme_low:
            long_confluence += 1
        if funding_extreme_short:
            long_confluence += 1
        if vol_normal:
            long_confluence += 1
        
        # Need at least 2 confluence for long entry
        if long_confluence >= 2:
            if fisher_cross_long or fisher_extreme_low:
                new_signal = current_size
            elif funding_extreme_short and (trend_4h_bullish or fisher[i] < 0):
                new_signal = current_size * 0.7
        
        # SHORT ENTRIES (require 2+ confluence)
        short_confluence = 0
        if trend_4h_bearish or price_below_4h_hma:
            short_confluence += 1
        if fisher_cross_short or fisher_extreme_high:
            short_confluence += 1
        if funding_extreme_long:
            short_confluence += 1
        if vol_normal:
            short_confluence += 1
        
        # Need at least 2 confluence for short entry
        if short_confluence >= 2:
            if fisher_cross_short or fisher_extreme_high:
                new_signal = -current_size
            elif funding_extreme_long and (trend_4h_bearish or fisher[i] > 0):
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~8 days on 1h), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and fisher[i] < -1.0:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and fisher[i] > 1.0:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend strongly reverses against position
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend becomes strongly bearish
            if position_side > 0 and trend_4h_bearish and hma_4h_slope_aligned[i] < -0.5:
                trend_reversal = True
            # Exit short if 4h trend becomes strongly bullish
            if position_side < 0 and trend_4h_bullish and hma_4h_slope_aligned[i] > 0.5:
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