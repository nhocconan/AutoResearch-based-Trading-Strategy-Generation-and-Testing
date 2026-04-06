#!/usr/bin/env python3
 """
 1d Williams Alligator + Elder Ray + Trend Filter
 Hypothesis: Alligator identifies trend direction, Elder Ray confirms strength,
             and weekly trend filter avoids counter-trend trades.
             Works in bull (buy when green above red, bullish Elder Ray, weekly bullish)
             and bear (sell when red above green, bearish Elder Ray, weekly bearish).
             Target: 75-200 total trades over 4 years.
 """

 import numpy as np
 import pandas as pd
 from mtf_data import get_htf_data, align_htf_to_ltf

 name = "1d_alligator_elder_ray_weekly_trend_v1"
 timeframe = "1d"
 leverage = 1.0

 def generate_signals(prices):
     n = len(prices)
     if n < 50:
         return np.zeros(n)
     
     # Price data
     high = prices['high'].values
     low = prices['low'].values
     close = prices['close'].values
     
     # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)
     # Smoothed with SMMA (similar to Wilder's smoothing)
     def smma(arr, period):
         res = np.full_like(arr, np.nan, dtype=float)
         if len(arr) < period:
             return res
         # First value is simple average
         res[period-1] = np.mean(arr[:period])
         # Subsequent values: (prev * (period-1) + current) / period
         for i in range(period, len(arr)):
             res[i] = (res[i-1] * (period-1) + arr[i]) / period
         return res
     
     jaw = smma(high, 13)  # Alligator Jaw (blue)
     teeth = smma(low, 8)   # Alligator Teeth (red)
     lips = smma(close, 5)  # Alligator Lips (green)
     
     # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
     def ema(arr, period):
         res = np.full_like(arr, np.nan, dtype=float)
         if len(arr) < period:
             return res
         multiplier = 2 / (period + 1)
         res[0] = arr[0]
         for i in range(1, len(arr)):
             res[i] = arr[i] * multiplier + res[i-1] * (1 - multiplier)
         return res
     
     ema13 = ema(close, 13)
     bull_power = high - ema13
     bear_power = low - ema13
     
     # Get weekly data for trend filter
     df_1w = get_htf_data(prices, '1w')
     close_1w = df_1w['close'].values
     
     # Weekly EMA50 for trend filter
     ema_1w = np.full(len(close_1w), np.nan, dtype=float)
     if len(close_1w) >= 50:
         ema_1w[49] = np.mean(close_1w[:50])
         for i in range(50, len(close_1w)):
             ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 48) / 50
     
     # Weekly trend: above EMA50 = bullish, below = bearish
     trend_1w = np.where(close_1w > ema_1w, 1, -1)
     
     # Align weekly trend to daily timeframe
     trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
     
     signals = np.zeros(n)
     position = 0  # 0: flat, 1: long, -1: short
     entry_price = 0.0
     bars_since_exit = 0
     
     # Start from warmup period (need enough data for Alligator and weekly alignment)
     start = 50
     
     for i in range(start, n):
         # Skip if required data not available
         if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
             np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
             np.isnan(trend_1w_aligned[i])):
             if position != 0:
                 signals[i] = position * 0.25
             else:
                 signals[i] = 0.0
             bars_since_exit += 1
             continue
         
         # Check exits
         if position == 1:  # long position
             # Exit: Alligator reverses (red above green) OR weekly trend turns bearish
             if (teeth[i] > lips[i] or trend_1w_aligned[i] == -1):
                 signals[i] = 0.0
                 position = 0
                 bars_since_exit = 0
             else:
                 signals[i] = 0.25
             bars_since_exit += 1
         elif position == -1:  # short position
             # Exit: Alligator reverses (green above red) OR weekly trend turns bullish
             if (lips[i] > teeth[i] or trend_1w_aligned[i] == 1):
                 signals[i] = 0.0
                 position = 0
                 bars_since_exit = 0
             else:
                 signals[i] = -0.25
             bars_since_exit += 1
         else:
             # Look for entries with minimum bars since exit
             if bars_since_exit >= 3:  # Prevent whipsaw
                 # Long: Green above Red (bullish alignment) AND Bull Power positive AND weekly bullish
                 if (lips[i] > teeth[i] and bull_power[i] > 0 and trend_1w_aligned[i] == 1):
                     signals[i] = 0.25
                     position = 1
                     entry_price = close[i]
                     bars_since_exit = 0
                 # Short: Red above Green (bearish alignment) AND Bear Power negative AND weekly bearish
                 elif (teeth[i] > lips[i] and bear_power[i] < 0 and trend_1w_aligned[i] == -1):
                     signals[i] = -0.25
                     position = -1
                     entry_price = close[i]
                     bars_since_exit = 0
                 else:
                     signals[i] = 0.0
                     bars_since_exit += 1
             else:
                 signals[i] = 0.0
                 bars_since_exit += 1
     
     return signals